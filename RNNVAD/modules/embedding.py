# -*- coding: utf-8 -*-

import torch
from torch import nn
import math

from utils import get_segment_ids, xavier_uniform_fan_in_


def initialize_embedding(weight, method):
    assert method in ("uniform1", "uniform2", "normal1", "normal2")
    if method == "uniform1":
        nn.init.uniform_(weight, a=-0.1, b=0.1)
    elif method == "normal1":
        nn.init.normal_(weight, std=0.02)
    elif method == "uniform2":
        nn.init.uniform_(weight, -math.sqrt(3), math.sqrt(3))
    else:
        nn.init.normal_(weight)


class Embeddings(nn.Module):
    def __init__(self,
                 word2syllable,
                 pretrained_embs,
                 *,
                 use_syllable_emb,
                 use_segment_emb,
                 embs_finetune_option,
                 PAD_token,
                 SEP_token,
                 dropout,
                 vocab_size,
                 embedding_dim,
                 output_dim,
                 nonfrozen_tokens,
                 segment_emb_relative,
                 sep_as_new_segment,
                 num_segments,
                 initialization_method):
        super(Embeddings, self).__init__()

        self.use_syllable_emb = use_syllable_emb
        self.use_segment_emb = use_segment_emb
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.segment_emb_relative = segment_emb_relative
        self.sep_as_new_segment = sep_as_new_segment
        assert pretrained_embs is None or isinstance(pretrained_embs, dict)
        pretrained_embs = dict() if pretrained_embs is None else pretrained_embs
        print("pretrained_embs: {}".format(" ".join(k for k, v in pretrained_embs.items() if v is not None)))

        self.word_emb = WordEmbedding(
            pretrained_embs.get("word", None),
            vocab_size=vocab_size,
            PAD_token=PAD_token,
            finetune=embs_finetune_option["word"],
            embedding_dim=embedding_dim["word"],
            output_dim=output_dim,
            nonfrozen_tokens=nonfrozen_tokens,
            initialization_method=initialization_method["word"])
        self.syllable_emb = SyllableEmbedding(
            word2syllable,
            pretrained_embs.get("syllable", None),
            use_syllable_emb=use_syllable_emb,
            finetune=embs_finetune_option["syllable"],
            embedding_dim=embedding_dim["syllable"],
            output_dim=output_dim,
            initialization_method=initialization_method["syllable"])
        self.segment_emb = MultiSegmentEmbedding(
            pretrained_embs.get("segment", None),
            use_segment_emb=use_segment_emb,
            embedding_dim=embedding_dim["segment"],
            output_dim=output_dim,
            num_segments=num_segments,
            relative=segment_emb_relative,
            SEP_token=SEP_token,
            sep_as_new_segment=sep_as_new_segment,
            finetune=embs_finetune_option["segment"],
            initialization_method=initialization_method["segment"])
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    # word_ids: [seq_len, batch_size]
    # segment_ids: [seq_len, batch_size]
    def forward(self,
                word_ids,
                align_pos,
                segment_ids=None,
                use_syllable_encodings=True,
                use_segment_encodings=True,
                scale_word_emb=False,
                segment_emb_restrict=True,
                use_segment_emb_cache=False):
        all_embs = {}
        all_embs["word"] = self.forward_word_emb(word_ids, scale=scale_word_emb)
        if use_syllable_encodings and self.use_syllable_emb:
            all_embs["syllable"] = self.syllable_emb(word_ids)
        if use_segment_encodings and self.use_segment_emb:
            all_embs["segment"] = self.segment_emb(
                word_ids,
                align_pos=align_pos,
                segment_ids=segment_ids,
                restrict=segment_emb_restrict,
                use_cache=use_segment_emb_cache)
        emb = torch.stack([v for k, v in all_embs.items() if k != "pos"], dim=-1).sum(dim=-1, keepdim=False)
        out = self.dropout(self.layer_norm(emb))
        return out

    def forward_word_emb(self, word_ids, scale=False):
        coeff = math.sqrt(self.output_dim) if scale else 1
        return self.word_emb(word_ids) * coeff

    def get_syllable_nums(self, word_ids):
        return self.syllable_emb.get_syllable_nums(word_ids)

    def get_segment_ids(self, *args, **kwargs):
        return self.segment_emb.get_segment_ids(*args, **kwargs)

    def clear_segment_emb_cache(self):
        self.segment_emb.clear_cache()

    def get_word_emb_weight(self):
        if not self.word_emb.weight_sharable:
            return None
        return self.word_emb.get_weight()

    def get_word_emb_out_proj_weight(self):
        return getattr(self.word_emb.out_proj, "weight", None)

    def get_word2syllable_buffer(self):
        return getattr(self.syllable_emb, "word2syllable", None)


class EmbeddingBase(nn.Module):
    def __init__(self, embedding_dim, output_dim, empty_base=False):
        super(EmbeddingBase, self).__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.empty_base = empty_base
        self.out_proj = nn.Identity()
        if not empty_base:
            if embedding_dim != output_dim:
                self.out_proj = nn.Linear(embedding_dim, output_dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        if self.empty_base:
            return
        if isinstance(self.out_proj, nn.Linear):
            xavier_uniform_fan_in_(self.out_proj.weight)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class WordEmbedding(EmbeddingBase):
    def __init__(self,
                 pretrained_weights,
                 *,
                 vocab_size,
                 PAD_token,
                 finetune,
                 embedding_dim,
                 output_dim,
                 initialization_method,
                 nonfrozen_tokens=()):
        super(WordEmbedding, self).__init__(embedding_dim, output_dim)

        assert PAD_token not in nonfrozen_tokens
        assert all(i < vocab_size for i in nonfrozen_tokens)
        self.vocab_size = vocab_size
        self.PAD_token = PAD_token
        self.finetune = finetune
        self.nonfrozen_tokens = sorted(set(nonfrozen_tokens))
        self.num_nonfrozen_tokens = len(self.nonfrozen_tokens)

        if pretrained_weights is None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.PAD_token)
            initialize_embedding(self.embedding.weight, initialization_method)
            self.embedding.weight.data[self.PAD_token] = 0.
        else:
            pretrained_weights = torch.as_tensor(pretrained_weights, dtype=torch.get_default_dtype())
            assert tuple(pretrained_weights.shape) == (self.vocab_size, self.embedding_dim), "{}, {}".format(
                tuple(pretrained_weights.shape), (self.vocab_size, self.embedding_dim))
            if self.num_nonfrozen_tokens == 0 or self.finetune:
                self.embedding = nn.Embedding.from_pretrained(pretrained_weights, padding_idx=self.PAD_token)
                self.embedding.weight.requires_grad = self.finetune
            else:
                frozen_tokens = sorted(set(range(self.vocab_size)) - set(self.nonfrozen_tokens))
                embid2wordid = torch.as_tensor(self.nonfrozen_tokens + frozen_tokens, dtype=torch.long)
                wordid2embid = torch.zeros(vocab_size, dtype=torch.long).index_copy_(
                    0, embid2wordid, torch.arange(vocab_size, dtype=torch.long))
                self.register_buffer("embid2wordid", embid2wordid)
                self.register_buffer("wordid2embid", wordid2embid)
                self.nonfrozen_emb = nn.Embedding.from_pretrained(pretrained_weights[self.nonfrozen_tokens])
                self.nonfrozen_emb.weight.requires_grad = True
                self.embedding = nn.Embedding.from_pretrained(pretrained_weights[frozen_tokens])
                self.embedding.weight.requires_grad = False

    def forward(self, input):
        if not hasattr(self, "nonfrozen_emb"):
            return self.out_proj(self.embedding(input))
        input = self.wordid2embid[input]
        embedded = torch.zeros(*input.shape, self.embedding_dim, dtype=torch.get_default_dtype(), device=input.device)
        mask = input >= self.num_nonfrozen_tokens
        embedded[mask] = self.embedding(input[mask] - self.num_nonfrozen_tokens)
        mask.logical_not_()
        embedded[mask] = self.nonfrozen_emb(input[mask])
        return self.out_proj(embedded)

    def get_weight(self):
        assert self.weight_sharable
        return self.embedding.weight

    @property
    def weight_sharable(self):
        return not hasattr(self, "nonfrozen_emb")


class SyllableEmbedding(EmbeddingBase):
    def __init__(self,
                 word2syllable,
                 pretrained_weights,
                 *,
                 use_syllable_emb,
                 finetune,
                 embedding_dim,
                 output_dim,
                 initialization_method):
        super(SyllableEmbedding, self).__init__(embedding_dim, output_dim, not use_syllable_emb)

        assert not use_syllable_emb or word2syllable is not None
        self.use_syllable_emb = use_syllable_emb

        if word2syllable is not None:
            self.num_classes = max(word2syllable) + 1
            self.register_buffer("word2syllable", torch.as_tensor(word2syllable, dtype=torch.long))
            if use_syllable_emb:
                if pretrained_weights is not None:
                    pretrained_weights = torch.as_tensor(pretrained_weights, dtype=torch.get_default_dtype())
                    assert tuple(pretrained_weights.shape) == (self.num_classes, self.embedding_dim)
                    self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
                    self.embedding.weight.requires_grad = finetune
                else:
                    self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
                    initialize_embedding(self.embedding.weight, initialization_method)

    def get_syllable_nums(self, input):
        return self.word2syllable[input] if hasattr(self, "word2syllable") else None

    def forward(self, input):
        syllable_nums = self.get_syllable_nums(input)
        return self.out_proj(self.embedding(syllable_nums)) if hasattr(self, "embedding") else None


class MultiSegmentEmbedding(EmbeddingBase):
    def __init__(self,
                 pretrained_weights,
                 *,
                 use_segment_emb,
                 embedding_dim,
                 output_dim,
                 relative,
                 SEP_token,
                 num_segments,
                 finetune,
                 sep_as_new_segment,
                 initialization_method):
        super(MultiSegmentEmbedding, self).__init__(embedding_dim, output_dim, not use_segment_emb)

        self.relative = relative
        self.num_segments = 2 if relative else num_segments
        self.SEP_token = SEP_token
        self.sep_as_new_segment = sep_as_new_segment

        if use_segment_emb:
            if pretrained_weights is not None:
                pretrained_weights = torch.as_tensor(pretrained_weights, dtype=torch.get_default_dtype())
                assert tuple(pretrained_weights.shape) == (self.num_segments, self.embedding_dim)
                self.embedding = nn.Embedding.from_pretrained(pretrained_weights)
                self.embedding.weight.requires_grad = finetune
            else:
                self.embedding = nn.Embedding(self.num_segments, self.embedding_dim)
                initialize_embedding(self.embedding.weight, initialization_method)
        self.cache = None

    # input : [seq_len, batch_size]
    def get_segment_ids(self, input, align_pos=0, use_cache=False, restrict=True):
        if not use_cache:
            segment_ids = get_segment_ids(input, self.SEP_token, align_pos, self.relative, self.sep_as_new_segment)
        else:
            if self.cache is None:
                segment_ids = self._update_cache(input)
            else:
                segment_ids = self._update_cache(input[-1:])
        assert segment_ids.shape == input.shape
        mask = segment_ids >= self.num_segments
        if mask.any():
            if restrict:
                raise RuntimeError("segment id is greater than the maximum embedding index")
            segment_ids = segment_ids.masked_fill(mask, self.num_segments - 1)
        return segment_ids

    # input : [seq_len, batch_size]
    def forward(self, input, align_pos=0, segment_ids=None, restrict=True, use_cache=False):
        assert not (input is None and segment_ids is None)
        assert not use_cache or align_pos == 0
        if segment_ids is None:
            segment_ids = self.get_segment_ids(input, align_pos, use_cache, restrict)
        else:
            assert segment_ids.shape == input.shape
        embedded = self.embedding(segment_ids)
        return self.out_proj(embedded)

    # input_step : [seq_len, batch_size]
    def _update_cache(self, input_step):
        if self.cache is None:
            ids = get_segment_ids(input_step, self.SEP_token, 0, self.relative, self.sep_as_new_segment)
        else:
            assert input_step.size(0) == 1, "only support step by step update"
            old, last_step = self.cache
            new_ids = old[-1].clone()
            step = input_step[-1] if self.sep_as_new_segment else last_step
            new_ids[step == self.SEP_token] += 1
            if self.relative:
                new_ids %= 2
            ids = torch.cat([old, new_ids.unsqueeze(0)], dim=0)
        self.clear_cache()
        self.cache = (ids, input_step[-1])
        return ids

    def clear_cache(self):
        del self.cache
        self.cache = None
