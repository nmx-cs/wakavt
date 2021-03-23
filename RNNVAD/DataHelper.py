# -*- coding:utf-8 -*-

import json
import re
import random
import os
import numpy as np
import math
from matplotlib import pyplot as plt
from collections import OrderedDict, defaultdict

from tools.jap_kana_tools import count_moras


class DataHelper(object):
    def __init__(self,
                 opts,
                 data_path,
                 keywords_path,
                 emb_path,
                 vocab_path,
                 word2id_path,
                 get_word2sylnum,
                 reform_data,
                 save_dir,
                 emb_save_dir):
        self.opts = opts
        for k in ("name2stoken", "special_tokens", "num_special_tokens", "latent_dim",
                  "valid_size", "test_size", "gen_size", "max_epoch_size", "embedding_dim"):
            setattr(self, k, getattr(self.opts, k))
        assert len(self.name2stoken) == len(self.special_tokens) == self.num_special_tokens
        self.min_len, self.max_len = self.opts.seq_len_range
        self.reform_data = reform_data
        self.get_word2sylnum = get_word2sylnum
        self.save_dir = save_dir
        self.emb_save_dir = emb_save_dir
        self.keywords_path = keywords_path

        if reform_data:
            self.all_data = DataHelper.load_from_json(filepath=data_path)
            self.all_data = self.filter_all_data()
            self.all_data = self.index_keywords()
            self.pretrained_emb = DataHelper.load_txt_embeddings(filepath=emb_path)
            self.vocab = DataHelper.load_txt_words(filepath=vocab_path)
            if self.vocab is None:
                self.vocab = list(self.pretrained_emb.keys())
            self.word2id = self.get_word2id()
            DataHelper.save_as_json(self.word2id, filename="word2id.json", filepath=word2id_path)
        else:
            self.all_data, self.vocab, self.pretrained_emb = None, None, None
            self.word2id = DataHelper.load_from_json(filename="word2id.json", filepath=word2id_path)
        self.id2word = {v: k for k, v in self.word2id.items()}

        if get_word2sylnum:
            self.word2sylnum = self.make_word2sylnum()

    def get_seq_len_range(self, data_list):
        min_len = min(min(len(t) for t, *_ in data) for data in data_list)
        max_len = max(max(len(t) for t, *_ in data) for data in data_list)
        return min_len, max_len

    def filter_all_data(self):
        filtered = []
        for t, *ki in self.all_data:
            if self.min_len <= len(t) <= self.max_len:
                filtered.append((t, *ki))
        all_data = filtered[:(self.max_epoch_size + self.valid_size + self.test_size)]
        print("num of samples after filtering:", len(all_data))
        print("seq_len_range(without SOS and EOS): {}".format(self.get_seq_len_range([all_data])))
        return all_data

    def index_keywords(self):
        if len(self.all_data[0]) == 1 or self.all_data[0][1] is None:
            assert all(len(k) == 0 or k[0] is None for t, *k in self.all_data)
            return [(t, None) for t, *_ in self.all_data]
        return [(t, t.index(k)) for t, k in self.all_data]

    def get_word2id(self):
        word2id = {v: getattr(self.opts, k) for k, v in self.name2stoken.items()}
        for token in self.special_tokens:
            assert token not in self.vocab
        for i, word in enumerate(self.vocab):
            word2id[word] = i + self.num_special_tokens
        return word2id

    def make_word2sylnum(self):
        word2sylnum = {k: count_moras(k) for k in self.word2id.keys()}
        for k in self.special_tokens:
            word2sylnum[k] = 0
        word2sylnum["<UNK>"] = max(word2sylnum.values()) + 1
        word2sylnum = [word2sylnum[self.id2word[i]] for i in range(len(self.id2word))]
        return word2sylnum

    def _data_partition(self, all_data, random_part=True):
        if random_part:
            random.shuffle(all_data)
        assert self.valid_size + self.test_size < len(all_data)
        valid_data = all_data[:self.valid_size]
        test_data = all_data[self.valid_size:(self.valid_size + self.test_size)]
        train_data = all_data[(self.valid_size + self.test_size):]
        return train_data, valid_data, test_data

    def data_len_dist(self, data, name="Train Data"):
        count = defaultdict(lambda : 0)
        for s in data:
            count[len(s)] += 1
        fig = plt.figure()
        plt.bar(x=count.keys(), height=count.values())
        plt.xlabel("length")
        plt.ylabel("quantity")
        plt.title("Length Distribution of " + name + " (without SOS and EOS)")
        plt.savefig(os.path.join(self.save_dir, "length_distribution_of_{}.jpg".format(
            name.lower().replace(" ", "_"))))
        plt.close(fig)

    def prepare_train_data(self, random_=True, load_pretrained_embs=True):
        if self.reform_data:
            embedding_matrix = self._make_emb_matrix(random_)
            train_data, valid_data, test_data = self._data_partition(self.all_data, random_)
            self.data_len_dist(next(zip(*train_data)))
            DataHelper.save_as_json(train_data, dirpath=self.save_dir, filename="train_data.json")
            DataHelper.save_as_json(valid_data, dirpath=self.save_dir, filename="valid_data.json")
            DataHelper.save_as_json(test_data, dirpath=self.save_dir, filename="test_data.json")
            DataHelper.save_as_bin(embedding_matrix, dirpath=self.emb_save_dir, filename="pretrained_word_emb.npy")
            print("Vocab size:", len(self.word2id))
            if hasattr(self, "word2sylnum"):
                print("Dim of sylnum one_hot:", max(self.word2sylnum) + 1)
        else:
            train_data = DataHelper.load_from_json(dirpath=self.save_dir, filename="train_data.json")
            valid_data = DataHelper.load_from_json(dirpath=self.save_dir, filename="valid_data.json")
            test_data = DataHelper.load_from_json(dirpath=self.save_dir, filename="test_data.json")
        pretrained_embs = {"word": None, "pos": None, "syllable": None, "segment": None}
        if load_pretrained_embs:
            word_emb = DataHelper.load_from_bin(dirpath=self.emb_save_dir, filename="pretrained_word_emb.npy", enable_default=True)
            pos_emb = DataHelper.load_from_bin(dirpath=self.emb_save_dir, filename="pretrained_pos_emb.npy", enable_default=True)
            segment_emb = DataHelper.load_from_bin(dirpath=self.emb_save_dir, filename="pretrained_segment_emb.npy", enable_default=True)
            syllable_emb = DataHelper.load_from_bin(dirpath=self.emb_save_dir, filename="pretrained_syllable_emb.npy", enable_default=True)
            pretrained_embs = {"word": word_emb, "pos": pos_emb, "syllable": syllable_emb, "segment": segment_emb}

        train_data = self.convert_to_ids(train_data)
        valid_data = self.convert_to_ids(valid_data)
        test_data = self.convert_to_ids(test_data)

        seq_len_range = self.get_seq_len_range([train_data, valid_data, test_data])
        data_config = {"vocab_size": len(self.word2id),
                       "epoch_size": len(train_data),
                       "valid_size": len(valid_data),
                       "test_size": len(test_data),
                       "seq_len_range": seq_len_range}
        self.opts.set_opts(data_config)

        return_dict = {"train": train_data, "valid": valid_data,
                       "test": test_data, "pretrained_embs": pretrained_embs,
                       "word2id": self.word2id, "id2word": self.id2word}
        if hasattr(self, "word2sylnum"):
            return_dict["word2sylnum"] = self.word2sylnum
        return return_dict

    def prepare_test_data(self):
        assert not self.reform_data
        test_data = DataHelper.load_from_json(dirpath=self.save_dir, filename="test_data.json")
        test_data = self.convert_to_ids(test_data)
        return_dict = {"test": test_data, "id2word": self.id2word}
        if hasattr(self, "word2sylnum"):
            return_dict["word2sylnum"] = self.word2sylnum
        return return_dict

    def prepare_gen_data(self, reform_gen_data, gen_size=None, latent_file_name="latent.npy", offset=0):
        if gen_size is not None:
            self.gen_size = gen_size
        latent_vectors = [None] * self.gen_size
        if self.opts.is_cvae:
            if not reform_gen_data:
                latent_vectors = DataHelper.load_from_bin(dirpath=self.save_dir, filename=latent_file_name)
                assert latent_vectors.shape[0] >= self.gen_size + offset
                latent_vectors = latent_vectors[offset:(self.gen_size + offset)]
            else:
                dims = [self.gen_size + offset, self.opts.gen_max_seq_len, self.latent_dim]
                latent_vectors = np.random.normal(0, 1, dims)
                DataHelper.save_as_bin(latent_vectors, dirpath=self.save_dir, filename=latent_file_name)
        if self.keywords_path.lower().endswith(".json"):
            keywords = DataHelper.load_from_json(filepath=self.keywords_path)
        else:
            assert self.keywords_path.lower().endswith(".txt")
            keywords = DataHelper.load_txt_words(filepath=self.keywords_path)
        if keywords is not None:
            if len(keywords) < self.gen_size:
                num_copies = math.ceil(self.gen_size / len(keywords))
                keywords = keywords * num_copies
            keywords = [self.word2id.get(token, self.word2id["<UNK>"]) for token in keywords][:self.gen_size]
        else:
            keywords = [None] * len(latent_vectors)
        return_dict = {"gen": list(zip(keywords, latent_vectors)), "id2word": self.id2word}
        if hasattr(self, "word2sylnum"):
            return_dict["word2sylnum"] = self.word2sylnum
        return return_dict

    def _make_emb_matrix(self, random_init=True):
        if self.pretrained_emb is None:
            print("WARNING: No pretrained_emb loaded, so embedding_matrix will not be made.")
            return None
        mean_l2_norm = np.mean([np.linalg.norm(vec, 2) for vec in self.pretrained_emb.values()])
        embedding_dim = len(next(iter(self.pretrained_emb.values())))
        assert self.embedding_dim["word"] == embedding_dim
        embedding_matrix = [self.pretrained_emb.get(self.id2word[i], None) for i in range(len(self.id2word))]
        missed_tokens = []
        for i in range(len(self.id2word)):
            if embedding_matrix[i] is None:
                missed_tokens.append(i)
        print("{} words not found in pretrained embeddings".format(len(missed_tokens)))
        for token_id in missed_tokens:
            if random_init:
                vec = np.random.randn(embedding_dim)
                vec = vec / np.linalg.norm(vec, 2) * mean_l2_norm
            else:
                vec = np.zeros(embedding_dim, dtype=np.float32)
            embedding_matrix[token_id] = vec
        embedding_matrix[self.word2id["<PAD>"]] = np.zeros(embedding_dim, dtype=np.float32)
        embeddings = np.asarray(embedding_matrix, dtype=np.float32)
        assert not np.isnan(embeddings).any()
        return embeddings

    def convert_to_ids(self, seq_list):
        def seq2ids(seq):
            return [[self.word2id.get(token, self.word2id["<UNK>"]) for token in seq[0]], seq[1]]
        return [seq2ids(seq) for seq in seq_list]

    @staticmethod
    def get_filepath(**kw):
        if kw["filepath"] is not None:
            if kw["filename"] is not None:
                return os.path.join(os.path.dirname(kw["filepath"]), kw["filename"])
            return kw["filepath"]
        else:
            assert kw["dirpath"] is not None
            return os.path.join(kw["dirpath"], kw["filename"])

    @classmethod
    def load_txt_embeddings(cls, *, filename=None, dirpath=None, filepath=None):
        filepath = cls.get_filepath(filename=filename, dirpath=dirpath, filepath=filepath)
        if not os.path.isfile(filepath):
            return None
        word2vec = OrderedDict()
        with open(filepath, "r", encoding="utf-8") as f:
            num_words, embedding_dim = next(f).strip().split(" ")
            num_words, embedding_dim = int(num_words), int(embedding_dim)
            for line in f:
                word, vec = line.strip().split(" ", 1)
                vec = np.fromstring(vec, dtype=np.float32, sep=" ")
                assert len(vec) == embedding_dim
                word2vec[word] = vec
        assert len(word2vec) == num_words
        return word2vec

    @classmethod
    def load_txt_words(cls, *, filename=None, dirpath=None, filepath=None):
        filepath = cls.get_filepath(filename=filename, dirpath=dirpath, filepath=filepath)
        if not os.path.isfile(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read().strip()
        return re.split(r"\s+", data)

    @classmethod
    def load_from_json(cls, *, filename=None, dirpath=None, filepath=None):
        filepath = cls.get_filepath(filename=filename, dirpath=dirpath, filepath=filepath)
        if not os.path.isfile(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @classmethod
    def save_as_json(cls, data, *, filename=None, dirpath=None, filepath=None):
        filepath = cls.get_filepath(filename=filename, dirpath=dirpath, filepath=filepath)
        dirpath = os.path.dirname(filepath)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load_from_bin(cls, *, filename=None, dirpath=None, filepath=None, enable_default=False):
        filepath = cls.get_filepath(filename=filename, dirpath=dirpath, filepath=filepath)
        try:
            return np.load(filepath)
        except Exception as e:
            if enable_default:
                return None
            raise e

    @classmethod
    def save_as_bin(cls, data, *, filename=None, dirpath=None, filepath=None):
        filepath = cls.get_filepath(filename=filename, dirpath=dirpath, filepath=filepath)
        dirpath = os.path.dirname(filepath)
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
        np.save(filepath, data)
