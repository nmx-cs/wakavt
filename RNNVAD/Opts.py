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
        # --- data setting ---
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
        self.model = "CVAD"                   # CVAD/CVAE/CLM
        self.dropout = 0.2
        self.embedding_dim = {"word": 128, "syllable": 128, "segment": 128}
        self.use_syllable_emb = False
        self.use_segment_emb = False
        self.segment_emb_relative = False
        self.sep_as_new_segment = True
        self.embeddings_to_finetune = ("word", "syllable", "segment")
        self.embedding_initialization = "uniform1"   # uniform1 / uniform2 / normal1 / normal2
        self.enc_num_layers = None
        self.enc_hidden_size = None
        self.enc_bidirectional = True
        self.latent_dim = 128
        self.latent_use_tanh = False
        self.latent_out_attach = True
        self.dec_num_layers = 3
        self.dec_hidden_size = 256
        self.use_logits_mask = True
        self.bwd_use_logits_mask = False
        self.use_layer_norm = True
        self.layer_norm_trainable = True
        self.detach_bwd_decoder_from_embedding = True

        # --- train/valid ---
        self.num_epochs = 20
        self.logging_rate = 100
        self.validation_rate = 1000
        self.logging_window = self.logging_rate
        self.validation_window = self.validation_rate
        self.train_sample_n = 1
        self.test_sample_n = 3
        self.checkpoint_rate = 5000
        self.ckpt_start_step = 100000
        self.ckpt_stop_step = None
        self.specific_checkpoints = (-1,)
        self.specific_validations = (-1,)
        self.ckpt_autosave_approaches = {"kld_range": (0.8, 1.6, -0.08, 10000, True, 10), "valid_best": ["elbo", "fwd_ppl", "fwd_ce"]}

        self.loss_mean = "steps"
        self.fwd_ce_coeff = 1.
        self.bwd_ce_coeff = 1.                  # 0 to disable

        self.kl_start = 0.0001
        self.kl_stop = 0.85
        self.kl_n_cycle = 1
        self.kl_ratio = 0.425
        self.kl_warmup = 0.0
        self.kl_increase_type = "linear"
        self.kl_annealing_type = "monotonic"

        self.use_bow_loss = True
        self.bow_coeff = 0.3
        self.bow_window = 5                  # None or int (indicate size of window)

        self.optimizer = "Adam"
        self.optim_params = {"lr": 0.0001, "weight_decay": 0.0}
        self.lr_decay = 1.
        self.lr_adjust_rate = 7000
        self.min_lr = 0.0001

        self.max_gradient_norm = 5.0

        # --- generation ---
        self.gen_mode = "greedy"                    # beam/greedy
        self.beam_width = 20
        self.length_norm = 0.0
        self.n_best = 20

    @property
    def is_cvae(self):
        return self.model in ("CVAE", "CVAD")

    @property
    def is_variational_autoregressive(self):
        return self.model == "CVAD"

    @property
    def need_bwd_out_proj_layer(self):
        return self.bwd_ce_coeff > 0

    @property
    def emb_out_dim(self):
        return self.embedding_dim["word"]

    @property
    def need_segment_ids(self):
        return self.use_segment_emb

    @property
    def need_remain_syllables(self):
        return self.use_logits_mask

    @property
    def fwd_need_remain_syllables(self):
        return self.need_remain_syllables

    @property
    def bwd_need_remain_syllables(self):
        return self.bwd_use_logits_mask

    @property
    def fwd_use_logits_mask(self):
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
    def loss_reduction(self):
        return self.loss_mean + "_mean"

    def check_args(self):
        for k in self.name2stoken.keys():
            assert getattr(self, k) in range(self.num_special_tokens)
        assert self.model in ("CVAD", "CVAE", "CLM")
        if self.model == "CVAE" or self.model == "CLM":
            raise ValueError("model {} is not implemented".format(self.model))
        assert isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, dict)
        if isinstance(self.embedding_dim, int):
            self.embedding_dim = {"word": self.embedding_dim,
                                  "syllable": self.embedding_dim,
                                  "segment": self.embedding_dim}
        assert set(self.embedding_dim.keys()) == {"word", "syllable", "segment"}
        if isinstance(self.embedding_initialization, str):
            self.embedding_initialization = dict.fromkeys(["word", "syllable", "segment"], self.embedding_initialization)
        else:
            assert isinstance(self.embedding_initialization, dict)
            assert set(self.embedding_initialization.keys()) == {"word", "syllable", "segment"}
        assert len(set(self.embeddings_to_finetune) - {"word", "syllable", "segment"}) == 0
        embs_finetune_option = dict.fromkeys(["word", "segment", "syllable"], False)
        embs_finetune_option.update(dict.fromkeys(self.embeddings_to_finetune, True))
        self.embeddings_to_finetune = embs_finetune_option
        assert self.n_best <= self.beam_width
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
