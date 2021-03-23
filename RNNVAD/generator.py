# -*- coding: utf-8 -*-

from data_loader import DataLoader
from model import build_model
from utils import cal_amount_of_params

import torch
import os
import json
import math
from itertools import chain


class Generator(object):
    def __init__(self, opts, data_dict, states_path):
        self.opts = opts

        self.train_states = torch.load(states_path, map_location="cpu")
        self.model = build_model(opts, data_dict.get("word2sylnum", None), None).to(opts.device)
        cal_amount_of_params(self.model)
        self.load_states()

        self.data = DataLoader(opts, "gen", None, None, None, data_dict["gen"], data_dict.get("word2sylnum", None))
        self.id2word = data_dict["id2word"]

        self.batch_size = opts.batch_size
        self.gen_size = opts.gen_size
        self.EOS_token = opts.EOS_token
        self.UNK_token = opts.UNK_token
        self.SPLIT_token = opts.SPLIT_token
        self.mode = opts.gen_mode
        self.cur_mode = opts.gen_mode
        self.n_best = opts.n_best
        self.beam_params = {"beam_width": opts.beam_width,
                            "length_norm": opts.length_norm,
                            "n_best": opts.n_best}

        self.filename_list = ["gen", os.path.basename(states_path).split(".")[0],
                              "mode_" + self.mode, "gen_size_" + str(self.gen_size)]

    def load_states(self):
        assert hasattr(self, "train_states")
        missing, unexpected = self.model.load_state_dict(self.train_states["model_state"], strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            print(
                "WARNING: when loading model states, there are some mismatching modules as follows, which may corrupt your model:")
            print(
                "Cannot find these modules in state_dict, which means parameters of these parts will not be updated:\n{}".format(
                    missing))
            print("Some modules in state_dict are unexpected, which will just be ignored:\n{}".format(unexpected))

    def _remove_EOS(self, gen_ids):
        res = [None] * len(gen_ids)
        for ind in range(len(gen_ids)):
            try:
                eos_idx = gen_ids[ind].index(self.EOS_token)
            except:
                eos_idx = None
            res[ind] = gen_ids[ind][:eos_idx]
        return res

    def _remove_UNK(self, gen_ids):
        return [ids for ids in gen_ids if self.UNK_token not in ids]

    def generate(self, mode=None):
        self.model.eval()
        if mode is None:
            mode = self.mode
        assert mode in ("greedy", "beam")
        self.cur_mode = mode

        gen_iterator = self.data.gen_iterator()
        num_batch = math.ceil(self.gen_size / self.batch_size)
        gen_ids = []
        with torch.no_grad():
            for _ in range(num_batch):
                keyword_ids, *normal_vector = next(gen_iterator)
                batch = self.model.generate(keyword_ids, *normal_vector, mode, self.beam_params)
                assert keyword_ids is not None
                prefix = torch.zeros(2, keyword_ids.size(0), dtype=torch.long, device=keyword_ids.device)
                prefix[0] = keyword_ids
                prefix[1].fill_(self.SPLIT_token)
                if mode == "beam":
                    prefix = prefix.repeat_interleave(self.opts.n_best, dim=1)
                batch = torch.cat([prefix, batch], dim=0)
                gen_ids.append(batch.t().tolist())

        return list(chain(*gen_ids))

    def _ids2seqs(self, gen_ids):
        return [[self.id2word[id] for id in sample] for sample in gen_ids]

    def process_gen_ids(self, gen_ids, remove_EOS=True, remove_UNK=False, only_top_beam=False):
        if remove_EOS:
            gen_ids = self._remove_EOS(gen_ids)
        if remove_UNK:
            gen_ids = self._remove_UNK(gen_ids)
        if only_top_beam and self.cur_mode == "beam":
            assert len(gen_ids) == self.gen_size * self.n_best
            gen_ids = [gen_ids[i] for i in range(0, len(gen_ids), self.n_best)]
        return self._ids2seqs(gen_ids)

    def save_results_as_json(self, results, save_dir, filename_no_extension):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filename = filename_no_extension + ".json"
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    def save_results_as_txt(self, results, save_dir, filename_no_extension):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filename = filename_no_extension + ".txt"
        with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
            for seq in results:
                seq = " ".join(seq)
                f.write("%s\n" % seq)

    def save_results(self, results, save_dir, remove_EOS=True, remove_UNK=False, only_top_beam=False):
        filename = self.filename_list.copy()
        if self.cur_mode == "beam":
            filename += ["{}_{}".format(k, v) for k, v in self.beam_params.items()]
        if remove_EOS:
            filename.append("re")
        if remove_UNK:
            filename.append("ru")
        if only_top_beam and self.cur_mode == "beam":
            filename.append("ot")
        filename = "-".join(filename)

        self.save_results_as_json(results, save_dir, filename)
        self.save_results_as_txt(results, save_dir, filename)
