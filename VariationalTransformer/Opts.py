# -*- coding: utf-8 -*-


import json
import os
import torch


class Opts(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, **kwargs):
        self.initialize()
        vars(self).update(kwargs)
        self.check_args()

    def __repr__(self):
        return self.dumps_opts()

    def initialize(self):
        # --- 数据相关 ---
        self.name2stoken = {
            "PAD_token": "<PAD>",
            "SOS_token": "<SOS>",
            "EOS_token": "<EOS>",
            "UNK_token": "<UNK>",
            "SEP_token": "−",
            "SPLIT_token": "<SPLITTER>"
        }
        stoken2id = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "−": 4,
            "<SPLITTER>": 5
        }
        # predefined tokens
        self.name2stoken.update({"SPECIAL{}_token".format(i): "<SPECIAL{}>".format(i) for i in range(5)})
        stoken2id.update({"<SPECIAL{}>".format(i): i + 6 for i in range(5)})
        for k, v in self.name2stoken.items():
            setattr(self, k, stoken2id[v])
        self.special_tokens = tuple(sorted(stoken2id.keys(), key=stoken2id.get))
        self.num_special_tokens = len(self.special_tokens)

        self.vocab_size = None
        self.seq_len_range = (0, 50)
        self.max_epoch_size = 2000000
        self.epoch_size = None
        self.valid_size = 10000
        self.test_size = 5000
        self.gen_size = 1000
        self.batch_size = 32

        # --- model setting ---
        self.model = "VT"                   # VT / T / L
        self.dropout = 0.1
        self.embedding_dim = {"word": 128, "pos": 128, "syllable": 128, "segment": 128}
        self.weight_typing = False
        self.pos_emb_type = "sinusoidal"        # sinusoidal / trainable
        self.use_syllable_emb = False
        self.use_segment_emb = False
        self.segment_emb_relative = False
        self.sep_as_new_segment = True
        self.embeddings_to_finetune = ("word", "pos", "syllable", "segment")
        self.embedding_initialization = "uniform1"   # uniform1 / uniform2 / normal1 / normal2
        self.use_gelu = True
        self.d_model = 128
        self.nhead = 4
        self.num_layers = 4
        self.dim_feedforward = 4 * self.d_model
        self.num_layers_before_latent = 2
        self.share_params_before_latent = False
        self.modules_to_share_before_latent = ("SA", "FF", "LN", "FFLN", "MG")      # FF/SA/LN/FFLN/MG
        self.keyword_approaches = ("khead",)
        self.latent_dim = self.d_model
        self.latent_use_tanh = False
        self.latent_out_attach = False
        self.use_logits_mask = True
        self.self_attn_levels = ["global", "sent", "segment"]
        self.self_attn_out_merge = "gmu_ori"
        self.self_attn_share_params = False
        self.hierarchical_before_latent = False
        self.hierarchical_after_latent = False

        # --- train/valid ---
        self.num_epochs = 10
        self.logging_rate = 100
        self.validation_rate = 1000
        self.logging_window = self.logging_rate
        self.validation_window = self.validation_rate
        self.train_sample_n = 1
        self.test_sample_n = 3
        self.checkpoint_rate = 1000
        self.ckpt_start_step = 40000
        self.ckpt_stop_step = None
        self.specific_checkpoints = (-1,)
        self.specific_validations = (-1,)
        self.ckpt_autosave_approaches = {"kld_range": (0.8, 1.6, -0.08, 10000, True, 10), "valid_best": ["elbo", "ppl", "ce"]}

        self.loss_mean = "steps"
        self.ce_coeff = 1.

        self.kl_start = 0.0001
        self.kl_stop = 1.0
        self.kl_n_cycle = 1
        self.kl_ratio = 1.0
        self.kl_warmup = 0.0
        self.kl_increase_type = "linear"
        self.kl_annealing_type = "monotonic"

        self.use_bow_loss = True
        self.bow_coeff = 1.
        self.bow_window = 5

        self.optimizer = "Adam"
        self.optim_params = {"lr": 0.0001, "weight_decay": 0.0}
        self.lr_decay = 1.
        self.lr_adjust_rate = 7000
        self.min_lr = 0.0001

        self.max_gradient_norm = 5.0

        # --- inference ---
        self.gen_mode = "greedy"                    # beam/greedy
        self.beam_width = 20
        self.length_norm = 0.0
        self.n_best = 20

    @property
    def CLS_token(self):
        return self.SPECIAL0_token

    @property
    def DROP_token(self):
        return self.SPECIAL1_token

    @property
    def KEYWORD_token(self):
        return self.SPECIAL2_token

    @property
    def is_cvae(self):
        return self.model in ("T", "VT")

    @property
    def is_variational_autoregressive(self):
        return self.model == "VT"

    @property
    def conditioned_by_keywords(self):
        if self.model == "VT":
            return "khead" in self.keyword_approaches
        return "khead" in self.keyword_approaches or "klatent" in self.keyword_approaches

    @property
    def need_segment_ids(self):
        return self.use_segment_emb or len(set(self.self_attn_levels) & {"segment", "sent"}) > 0

    @property
    def need_remain_syllables(self):
        return self.use_logits_mask

    @property
    def pad_max_seq_len(self):
        return self.seq_len_range[-1] + 1

    @property
    def emb_max_seq_len(self):
        return self.pad_max_seq_len

    @property
    def gen_max_seq_len(self):
        return self.pad_max_seq_len

    @property
    def share_SA_before_latent(self):
        return self.share_params_before_latent and "SA" in self.modules_to_share_before_latent

    @property
    def share_LN_before_latent(self):
        return self.share_params_before_latent and "LN" in self.modules_to_share_before_latent

    @property
    def share_MG_before_latent(self):
        return self.share_params_before_latent and "MG" in self.modules_to_share_before_latent

    @property
    def share_FF_before_latent(self):
        return self.share_params_before_latent and "FF" in self.modules_to_share_before_latent

    @property
    def share_FFLN_before_latent(self):
        return self.share_params_before_latent and "FFLN" in self.modules_to_share_before_latent

    @property
    def hierarchical_model(self):
        return self.hierarchical_before_latent or self.hierarchical_after_latent

    @property
    def num_self_attn_per_layer_before_latent(self):
        if self.hierarchical_model:
            return 1
        return len(self.self_attn_levels)

    @property
    def num_self_attn_per_layer_after_latent(self):
        if self.hierarchical_model:
            return 1
        return len(self.self_attn_levels)

    @property
    def num_self_attn_per_layer(self):
        assert self.model == "L"
        if self.hierarchical_model:
            return 1
        return len(self.self_attn_levels)

    @property
    def loss_reduction(self):
        return self.loss_mean + "_mean"

    @property
    def kld_loss_reduction(self):
        return "batch_mean" if self.model == "T" else self.loss_reduction

    def check_args(self):
        for k in self.name2stoken.keys():
            assert getattr(self, k) in range(self.num_special_tokens)
        assert self.model in ("VT", "T", "L")
        assert isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, dict)
        if isinstance(self.embedding_dim, int):
            self.embedding_dim = {"word": self.embedding_dim,
                                  "pos": self.embedding_dim,
                                  "syllable": self.embedding_dim,
                                  "segment": self.embedding_dim}
        assert set(self.embedding_dim.keys()) == {"word", "pos", "syllable", "segment"}
        if isinstance(self.embedding_initialization, str):
            self.embedding_initialization = dict.fromkeys(["word", "pos", "syllable", "segment"], self.embedding_initialization)
        else:
            assert isinstance(self.embedding_initialization, dict)
            assert set(self.embedding_initialization.keys()) == {"word", "pos", "syllable", "segment"}
        assert len(set(self.embeddings_to_finetune) - {"word", "pos", "syllable", "segment"}) == 0
        embs_finetune_option = dict.fromkeys(["word", "segment", "syllable", "pos"], False)
        embs_finetune_option.update(dict.fromkeys(self.embeddings_to_finetune, True))
        self.embeddings_to_finetune = embs_finetune_option
        assert self.n_best <= self.beam_width
        assert len(set(self.modules_to_share_before_latent) - {"FF", "SA", "LN", "FFLN", "MG"}) == 0
        assert len(set(self.keyword_approaches) - {"khead", "ktoken", "klatent"}) == 0
        assert "ktoken" not in self.keyword_approaches or "khead" in self.keyword_approaches
        assert self.model != "L" or "khead" in self.keyword_approaches
        assert len(set(self.self_attn_levels) - {"segment", "sent", "global"}) == 0
        assert "global" in self.self_attn_levels
        if self.hierarchical_model:
            assert len(self.self_attn_levels) == 3
            if self.hierarchical_before_latent:
                assert self.num_layers_before_latent == 3
            if self.hierarchical_after_latent:
                assert self.num_layers - self.num_layers_before_latent == 3
        assert not self.latent_out_attach or self.model == "VT"
        assert self.loss_mean in ("batch", "steps")

    def to_dict(self):
        return vars(self).copy()

    def set_opts(self, options):
        assert isinstance(options, dict)
        vars(self).update(options)
        self.check_args()

    def _check_match(self, opts):
        missing = set(vars(self).keys()) - set(opts.keys())
        unexpected = set(opts.keys()) - set(vars(self).keys())
        if len(missing) > 0:
            print("Found missing options when loading opts: {}".format({k: getattr(self, k) for k in missing}))
        if len(unexpected) > 0:
            print("Found unexpected options when loading opts: {}".format({k: opts.get(k) for k in unexpected}))

    def loads_opts(self, json_str):
        opts = json.loads(json_str)
        assert isinstance(opts, dict)
        self._check_match(opts)
        vars(self).update(opts)
        self.check_args()

    def dumps_opts(self):
        return json.dumps(vars(self), indent=4, ensure_ascii=False)

    def load_opts(self, path=None):
        if path is None:
            filepath = os.path.abspath(__file__)
            path = os.path.join(os.path.dirname(filepath), "opts.json")
        with open(path, "r", encoding="utf-8") as f:
            opts = json.load(f)
        assert isinstance(opts, dict)
        self._check_match(opts)
        vars(self).update(opts)
        self.check_args()

    def dump_opts(self, path=None):
        if path is None:
            filepath = os.path.abspath(__file__)
            path = os.path.join(os.path.dirname(filepath), "opts.json")
        else:
            dirname = os.path.dirname(path)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vars(self), f, indent=4, ensure_ascii=False)
