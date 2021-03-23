# -*- coding: utf-8 -*-


import random
import torch
from torch.nn.utils import rnn as rnn_utils
from utils import get_segment_ids, get_remain_syllables


class DataLoader(object):
    def __init__(self,
                 opts,
                 mode,
                 train_data,
                 valid_data,
                 test_data,
                 gen_data,
                 word2syllable=None,
                 prepare_test=False,
                 shuffle=True,
                 bucketing=False):
        self.opts = opts
        self.batch_size = opts.batch_size
        self.EOS_token = opts.EOS_token
        self.PAD_token = opts.PAD_token
        self.SEP_token = opts.SEP_token
        if mode != "gen" and word2syllable is not None:
            self.word2syllable = torch.as_tensor(word2syllable, dtype=torch.long)
        else:
            self.word2syllable = None

        if mode == "train":
            assert train_data is not None and valid_data is not None
            assert self.opts.epoch_size == len(train_data)
            assert self.opts.valid_size == len(valid_data)
            self.epoch_size = len(train_data)
            self.valid_size = len(valid_data)
            self.train_data = self.preprocess_data(train_data, shuffle, bucketing)
            self.valid_data = self.preprocess_data(valid_data, False, bucketing)
            if prepare_test:
                assert test_data is not None
                assert self.opts.test_size == len(test_data)
                self.test_size = len(test_data)
                self.test_data = self.preprocess_data(test_data, False, bucketing)

        elif mode == "test":
            assert test_data is not None
            assert self.opts.test_size == len(test_data)
            self.test_size = len(test_data)
            self.test_data = self.preprocess_data(test_data, False, bucketing)

        elif mode == "gen":
            assert gen_data is not None
            assert self.opts.gen_size == len(gen_data)
            self.gen_data = self._get_batches(gen_data, False, False)
            self.gen_size = self.opts.gen_size
        else:
            raise ValueError("Param mode should be 'train', 'test' or 'gen'.")

        print("DataLoader Ready")
        if hasattr(self, "train_data"):
            print("train_data : %d" % self.epoch_size)
        for type_ in ("valid", "test", "gen"):
            data_name = type_ + "_data"
            if hasattr(self, data_name):
                print("%s : %d" % (data_name, getattr(self, type_ + "_size")))

    def _append_eos(self, data):
        for t, *_ in data:
            t.append(self.EOS_token)

    def _get_batches(self, data, shuffle=True, bucketing=False):
        num_samples = len(data)
        if shuffle and not bucketing:
            random.shuffle(data)
        if bucketing:
            data.sort(key=lambda x: len(x[0]))
        rptr = num_samples - num_samples % self.batch_size
        batches = [data[i:(i + self.batch_size)] for i in range(0, rptr, self.batch_size)]
        if rptr < len(data):
            batches.append(data[rptr:])
        if shuffle and bucketing:
            random.shuffle(batches)
        return batches

    def _pad_batch_sequence(self, batch):
        batch = [torch.as_tensor(seq, dtype=torch.long) for seq in batch]
        padded = rnn_utils.pad_sequence(batch, batch_first=False, padding_value=self.PAD_token)
        return padded

    def preprocess_data(self, data, shuffle, bucketing):
        self._append_eos(data)
        data = self._get_batches(data, shuffle=shuffle, bucketing=bucketing)
        res = []
        for batch in data:
            texts, keyword_pos = zip(*batch)
            lens = torch.as_tensor([len(s) for s in texts], dtype=torch.long)
            if keyword_pos[0] is None:
                keywords = keyword_pos = None
            else:
                keywords = torch.as_tensor([t[p] for t, p in batch], dtype=torch.long)
                keyword_pos = torch.as_tensor(keyword_pos, dtype=torch.long)
            texts = self._pad_batch_sequence(texts)
            res.append([texts, lens, keywords, keyword_pos])
            if self.opts.need_segment_ids:
                segment_ids = get_segment_ids(texts, self.SEP_token, 0, relative=self.opts.segment_emb_relative,
                                              sep_as_new_segment=self.opts.sep_as_new_segment)
                res[-1].append(segment_ids)
            if self.opts.need_remain_syllables:
                remain_syllables = get_remain_syllables(self.word2syllable, self.SEP_token, decoder_target=texts)
                res[-1].append(remain_syllables)
        return res

    def to_device(self, t):
        if not isinstance(t, torch.Tensor):
            return t
        return t.to(self.opts.device)

    def data_iterator(self, type_, start=0, max_epochs=None):
        if not hasattr(self, type_ + "_data"):
            return None
        data = getattr(self, type_ + "_data")
        num_epochs = 0

        i = start
        while max_epochs is None or num_epochs < max_epochs:
            texts, lens, keywords, keyword_pos, *others = data[i]
            assert len(others) <= 2
            if len(others) == 1 and self.opts.need_remain_syllables:
                others.insert(0, None)
            yield (self.to_device(texts), self.to_device(lens), self.to_device(keywords), self.to_device(keyword_pos),
                   *[self.to_device(t) for t in others])
            i += 1
            if i >= len(data):
                num_epochs += 1
                i = 0

    def train_iterator(self, start=0):
        return self.data_iterator("train", start)

    def valid_iterator(self):
        return self.data_iterator("valid")

    def test_iterator(self):
        if hasattr(self, "test_data"):
            return self.data_iterator("test")
        else:
            return None

    def gen_iterator(self):
        i = 0
        while True:
            keyword_ids, latent_vector = zip(*self.gen_data[i])
            if latent_vector[0] is None:
                latent_vector = None
            else:
                latent_vector = torch.as_tensor(latent_vector, dtype=torch.get_default_dtype(), device=self.opts.device)
                if self.opts.is_variational_autoregressive:
                    assert latent_vector.dim() == 3
                    latent_vector = latent_vector.transpose(0, 1).contiguous()
                else:
                    assert latent_vector.dim() in (2, 3)
                    if latent_vector.dim() == 3:
                        latent_vector = latent_vector[:, 0]
            if keyword_ids[0] is None:
                keyword_ids = None
            else:
                keyword_ids = torch.as_tensor(keyword_ids, dtype=torch.long, device=self.opts.device)
            vars = [keyword_ids]
            if self.opts.is_cvae:
                vars.append(latent_vector)
            yield vars
            i = (i + 1) % len(self.gen_data)


class CVADDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(CVADDataLoader, self).__init__(*args, **kwargs)

    def preprocess_data(self, data, shuffle, bucketing):
        self._append_eos(data)
        data = self._get_batches(data, shuffle=shuffle, bucketing=bucketing)
        res = []
        for batch in data:
            texts, keyword_pos = zip(*batch)
            lens = torch.as_tensor([len(s) for s in texts], dtype=torch.long)
            if keyword_pos[0] is None:
                keywords = keyword_pos = None
            else:
                keywords = torch.as_tensor([t[p] for t, p in batch], dtype=torch.long)
                keyword_pos = torch.as_tensor(keyword_pos, dtype=torch.long)
            fwd_tgt = self._pad_batch_sequence(texts)
            bwd_src = fwd_tgt.flip(0)
            res.append([[fwd_tgt, bwd_src], lens, keywords, keyword_pos])
            if self.opts.need_segment_ids:
                fwd_seg_ids = get_segment_ids(fwd_tgt, self.SEP_token, 0, relative=self.opts.segment_emb_relative,
                                              sep_as_new_segment=self.opts.sep_as_new_segment)
                bwd_seg_ids = fwd_seg_ids.flip(0)
                res[-1].append([fwd_seg_ids, bwd_seg_ids])
            if self.opts.fwd_need_remain_syllables or self.opts.bwd_need_remain_syllables:
                fwd_rem_syl = None
                if self.opts.fwd_need_remain_syllables:
                    fwd_rem_syl = get_remain_syllables(self.word2syllable, self.SEP_token, decoder_target=fwd_tgt)
                bwd_rem_syl = None
                if self.opts.bwd_need_remain_syllables:
                    bwd_rem_syl = get_remain_syllables(self.word2syllable, self.SEP_token, decoder_input=bwd_src)
                res[-1].append([fwd_rem_syl, bwd_rem_syl])
        return res

    def to_device(self, t):
        if isinstance(t, list):
            return [super(CVADDataLoader, self).to_device(ti) for ti in t]
        return super(CVADDataLoader, self).to_device(t)
