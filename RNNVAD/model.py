# -*- coding: utf-8 -*-

from modules.embedding import Embeddings
from modules.common_layers import LatentLayer, LogitsMaskLayer
from modules.gru import StackedGRUCell
from modules.beam import Beam, step_batch_beams
from utils import xavier_uniform_fan_in_, bidirectional_padding, get_triangle_bool_mask

import torch
from torch import nn
from itertools import chain


def build_model(opts, word2syllable, pretrained_emb_weights):
    if opts.model == "CVAD":
        return CVAD(opts, word2syllable, pretrained_emb_weights)
    elif opts.model == "CVAE":
        raise ValueError("Not implemented")
    elif opts.model == "CLM":
        raise ValueError("Not implemented")
    else:
        raise ValueError("Unknown model type: {}".format(opts.model))


def _build_embeddings(opts, word2syllable, pretrained_emb_weights):
    nonfrozen_tokens = [getattr(opts, token) for token in opts.name2stoken.keys() if token != "PAD_token"]
    return Embeddings(
        word2syllable,
        pretrained_embs=pretrained_emb_weights,
        use_syllable_emb=opts.use_syllable_emb,
        use_segment_emb=opts.use_segment_emb,
        embs_finetune_option=opts.embeddings_to_finetune,
        PAD_token=opts.PAD_token,
        SEP_token=opts.SEP_token,
        dropout=opts.dropout,
        vocab_size=opts.vocab_size,
        embedding_dim=opts.embedding_dim,
        output_dim=opts.emb_out_dim,
        nonfrozen_tokens=nonfrozen_tokens,
        segment_emb_relative=opts.segment_emb_relative,
        sep_as_new_segment=opts.sep_as_new_segment,
        num_segments=5,
        initialization_method=opts.embedding_initialization)


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()

    def _reset_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    def get_bow_tgt(self, tgt):
        if not self.opts.is_variational_autoregressive:
            return tgt
        seq_len, batch_size = tgt.size()
        mask = get_triangle_bool_mask(seq_len, device=tgt.device)
        if self.opts.bow_window is None:
            bow_tgt = tgt.unsqueeze(1).repeat(1, seq_len, 1).masked_fill_(mask.unsqueeze(-1), self.opts.PAD_token)
        else:
            padding = torch.full((self.opts.bow_window - 1,), fill_value=self.opts.PAD_token, dtype=tgt.dtype, device=tgt.device)
            tgt = torch.cat([tgt, padding.unsqueeze(-1).expand(-1, batch_size)], dim=0)
            expanded_tgt = tgt.unsqueeze(0).expand(seq_len, -1, -1)
            parallelogram_mask = torch.ones(*expanded_tgt.shape[:-1], dtype=torch.bool, device=tgt.device)
            parallelogram_mask[:, -seq_len:].masked_fill_(mask, 0)
            parallelogram_mask[:, :seq_len].masked_fill_(mask.t(), 0)
            bow_tgt = expanded_tgt[parallelogram_mask].view(seq_len, self.opts.bow_window, batch_size)
            bow_tgt = bow_tgt.transpose(0, 1).contiguous()
        return bow_tgt

    @staticmethod
    def expand_tgt(tgt, sample_n):
        if sample_n > 1:
            tgt = tgt.repeat(1, sample_n)
        return tgt


class VADLatentModule(nn.Module):
    def __init__(self, hidden_size, latent_dim, use_tanh):
        super(VADLatentModule, self).__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.use_tanh = use_tanh

        mlp_size = (hidden_size + latent_dim) // 2
        self.prior_net = LatentLayer(hidden_size, mlp_size, latent_dim, use_tanh)
        self.recognition_net = LatentLayer(2 * hidden_size, mlp_size, latent_dim, use_tanh)

    # forward_hidden: [sample_n * batch_size, hidden_size]
    # backward_hidden: [sample_n * batch_size, hidden_size]
    # normal_vector: [sample_n * batch_size, latent_dim]
    def forward_train_path(self, forward_hidden, backward_hidden, normal_vector=None):
        assert forward_hidden.shape == backward_hidden.shape
        prior_input = forward_hidden.view(-1, self.hidden_size)
        recognition_input = torch.cat([forward_hidden, backward_hidden], dim=-1).view(-1, 2 * self.hidden_size)
        if normal_vector is not None:
            normal_vector = normal_vector.view(-1, self.latent_dim)
        mu_p, log_var_p, _ = self.prior_net(prior_input, sample_n=1, reparam=False)
        mu_r, log_var_r, latent_vector = self.recognition_net(recognition_input, normal_vector, sample_n=1)
        return mu_p, log_var_p, mu_r, log_var_r, latent_vector.squeeze(0)

    def forward_gen_path(self, forward_hidden, normal_vector=None):
        *_, latent_vector = self.prior_net(forward_hidden, normal_vector, sample_n=1)
        return latent_vector.squeeze(0)


class CVAD(ModelBase):
    def __init__(self, opts, word2syllable, pretrained_emb_weights):
        super(ModelBase, self).__init__()

        self.opts = opts
        self.PAD_token = opts.PAD_token
        self.SOS_token = opts.SOS_token
        self.EOS_token = opts.EOS_token
        self.UNK_token = opts.UNK_token
        self.SEP_token = opts.SEP_token

        self.embedding = _build_embeddings(opts, word2syllable, pretrained_emb_weights)

        self.latent_module = VADLatentModule(opts.dec_hidden_size, opts.latent_dim, opts.latent_use_tanh)
        if opts.use_bow_loss:
            self.bow_proj_layer = nn.Linear(opts.latent_dim + opts.dec_hidden_size, opts.vocab_size)

        self.emb2states_fwd = nn.Linear(opts.emb_out_dim, opts.dec_num_layers * opts.dec_hidden_size)
        self.emb2states_bwd = nn.Linear(opts.emb_out_dim, opts.dec_num_layers * opts.dec_hidden_size)

        self.fwd_decoder = StackedGRUCell(opts.emb_out_dim + opts.latent_dim, opts.dec_hidden_size, opts.dec_num_layers,
                                          opts.dropout, opts.use_layer_norm, opts.layer_norm_trainable)
        self.bwd_decoder = StackedGRUCell(opts.emb_out_dim, opts.dec_hidden_size, opts.dec_num_layers,
                                          opts.dropout, opts.use_layer_norm, opts.layer_norm_trainable)

        out_dim = opts.dec_hidden_size
        if opts.latent_out_attach:
            out_dim += opts.latent_dim
        self.fwd_out_proj = nn.Linear(out_dim, opts.vocab_size, bias=False)
        if opts.need_bwd_out_proj_layer:
            self.bwd_out_proj = nn.Linear(opts.dec_hidden_size, opts.vocab_size, bias=False)

        if opts.fwd_use_logits_mask:
            self.fwd_logits_mask_layer = LogitsMaskLayer(
                self.embedding.get_word2syllable_buffer(), opts.SEP_token, opts.UNK_token)
        if opts.bwd_use_logits_mask:
            self.bwd_logits_mask_layer = LogitsMaskLayer(
                self.embedding.get_word2syllable_buffer(), opts.SEP_token, opts.UNK_token)

        self._reset_parameters()

    def _reset_parameters(self):
        if hasattr(self, "bow_proj_layer"):
            xavier_uniform_fan_in_(self.bow_proj_layer.weight)
            nn.init.zeros_(self.bow_proj_layer.bias)
        xavier_uniform_fan_in_(self.emb2states_fwd.weight)
        xavier_uniform_fan_in_(self.emb2states_bwd.weight)
        nn.init.zeros_(self.emb2states_fwd.bias)
        nn.init.zeros_(self.emb2states_bwd.bias)
        if hasattr(self, "fwd_out_proj"):
            xavier_uniform_fan_in_(self.fwd_out_proj.weight)
        if hasattr(self, "bwd_out_proj"):
            xavier_uniform_fan_in_(self.bwd_out_proj.weight)

    def _init_states(self, keyword_embs, direction):
        assert direction in ("fwd", "bwd")
        batch_size = keyword_embs.size(0)
        if direction == "fwd":
            hidden = self.emb2states_fwd(keyword_embs)
        else:
            if self.opts.detach_bwd_decoder_from_embedding:
                keyword_embs = keyword_embs.detach()
            hidden = self.emb2states_bwd(keyword_embs)
        hidden = hidden.view(batch_size, self.opts.dec_num_layers, self.opts.dec_hidden_size)
        return hidden.transpose(0, 1).contiguous()

    def _forward_bwd_decoder(self, bwd_input, bwd_segment_ids, bwd_remain_syllables, initial_hidden):
        seq_len, batch_size = bwd_input.size()
        device = bwd_input.device
        embedded = self.embedding(bwd_input, 0, segment_ids=bwd_segment_ids)
        if self.opts.detach_bwd_decoder_from_embedding:
            embedded = embedded.detach()
        bwd_last_layer_states = []
        prev_hidden = initial_hidden
        for step in range(seq_len):
            cur_hidden = self.bwd_decoder(embedded[step], prev_hidden)
            pad_indexes = torch.arange(batch_size, device=device)[bwd_input[step] == self.opts.PAD_token]
            cur_hidden = cur_hidden.index_copy(1, pad_indexes, prev_hidden[:, pad_indexes])
            bwd_last_layer_states.append(cur_hidden[-1])
            prev_hidden = cur_hidden
        bwd_last_layer_hidden = torch.stack(bwd_last_layer_states, dim=0)
        logits = None
        if hasattr(self, "bwd_out_proj"):
            logits = self.bwd_out_proj(bwd_last_layer_hidden)
            if self.opts.bwd_use_logits_mask:
                logits = self.bwd_logits_mask_layer(
                    logits,
                    remain_syllables=bwd_remain_syllables,
                    decoder_input=bwd_input,
                    sample_n_to_check=1)
        return logits, bwd_last_layer_hidden

    def _forward_fwd_decoder(self, fwd_input, fwd_segment_ids, fwd_remain_syllables, initial_hidden, bwd_hidden, sample_n):
        fwd_last_layer_states = []
        mu_p_list = []
        log_var_p_list = []
        mu_r_list = []
        log_var_r_list = []
        latent_vector_list = []

        embedded = self.embedding(fwd_input, 0, segment_ids=fwd_segment_ids)
        prev_hidden = initial_hidden
        if sample_n > 1:
            embedded = embedded.repeat(1, sample_n, 1)
            prev_hidden = prev_hidden.repeat(1, sample_n, 1)
            bwd_hidden = bwd_hidden.repeat(1, sample_n, 1)

        for step in range(fwd_input.size(0)):
            mu_p, log_var_p, mu_r, log_var_r, z = self.latent_module.forward_train_path(
                prev_hidden[-1], bwd_hidden[-(step + 1)])
            cur_hidden = self.fwd_decoder(torch.cat([embedded[step], z], dim=-1), prev_hidden)
            fwd_last_layer_states.append(cur_hidden[-1])
            prev_hidden = cur_hidden

            mu_p_list.append(mu_p)
            log_var_p_list.append(log_var_p)
            mu_r_list.append(mu_r)
            log_var_r_list.append(log_var_r)
            latent_vector_list.append(z)

        fwd_last_layer_hidden = torch.stack(fwd_last_layer_states, dim=0)
        latent_vector = torch.stack(latent_vector_list, dim=0)
        out_proj_inp = fwd_last_layer_hidden
        if self.opts.latent_out_attach:
            out_proj_inp = torch.cat([fwd_last_layer_hidden, latent_vector], dim=-1)
        logits = self.fwd_out_proj(out_proj_inp)
        if self.opts.fwd_use_logits_mask:
            logits = self.fwd_logits_mask_layer(
                logits,
                remain_syllables=fwd_remain_syllables,
                decoder_input=fwd_input,
                sample_n_to_check=sample_n)

        mu_p = torch.stack(mu_p_list, dim=0)
        log_var_p = torch.stack(log_var_p_list, dim=0)
        mu_r = torch.stack(mu_r_list, dim=0)
        log_var_r = torch.stack(log_var_r_list, dim=0)

        return logits, fwd_last_layer_hidden, latent_vector, mu_p, log_var_p, mu_r, log_var_r

    def forward(self, inputs, keyword_ids, segment_ids=None, remain_syllables=None, mode="train"):
        assert mode in ("train", "valid", "test")
        if mode != "train":
            assert not self.training
        else:
            assert self.training

        fwd_tgt, bwd_inp = inputs
        sos = torch.full((fwd_tgt.size(1),), fill_value=self.SOS_token, dtype=torch.long, device=fwd_tgt.device)
        fwd_inp = torch.cat([sos.unsqueeze(0), fwd_tgt[:-1]], dim=0)
        bwd_tgt = fwd_inp.flip(0)

        if self.opts.need_segment_ids:
            if segment_ids is None:
                fwd_seg_ids = self.embedding.get_segment_ids(fwd_inp)
                bwd_seg_ids = self.embedding.get_segment_ids(bwd_inp)
            else:
                fwd_seg_ids, bwd_seg_ids = segment_ids
                fwd_seg_ids = torch.cat([torch.zeros_like(fwd_seg_ids[:1]), fwd_seg_ids[:-1]], dim=0)
        else:
            fwd_seg_ids = bwd_seg_ids = None
        fwd_rem_syls = bwd_rem_syls = None
        if self.opts.fwd_need_remain_syllables or self.opts.bwd_need_remain_syllables:
            if remain_syllables is None:
                if self.opts.fwd_need_remain_syllables:
                    fwd_rem_syls = self.logits_mask_layer.get_remain_syllables(fwd_inp)
                if self.opts.bwd_need_remain_syllables:
                    bwd_rem_syls = self.logits_mask_layer.get_remain_syllables(bwd_inp)
            else:
                fwd_rem_syls, bwd_rem_syls = remain_syllables

        keyword_embs = self.embedding.forward_word_emb(keyword_ids)
        fwd_initial_states = self._init_states(keyword_embs, "fwd")
        bwd_initial_states = self._init_states(keyword_embs, "bwd")

        bwd_logits, bwd_hidden = self._forward_bwd_decoder(bwd_inp, bwd_seg_ids, bwd_rem_syls, bwd_initial_states)

        sample_n = self.opts.train_sample_n if mode == "train" else self.opts.test_sample_n
        fwd_logits, fwd_last_layer_hidden, latent_vector, mu_p, log_var_p, mu_r, log_var_r = self._forward_fwd_decoder(
            fwd_inp, fwd_seg_ids, fwd_rem_syls, fwd_initial_states, bwd_hidden, sample_n)

        bow_logits = None
        if self.opts.use_bow_loss:
            bow_inp = torch.cat([latent_vector, fwd_last_layer_hidden], dim=-1)
            bow_logits = self.bow_proj_layer(bow_inp)
            expand_dim = bow_logits.size(0) if self.opts.bow_window is None else self.opts.bow_window
            bow_logits = bow_logits.unsqueeze(0).expand(expand_dim, -1, -1, -1)

        fwd_tgt = self.expand_tgt(fwd_tgt, sample_n)

        return (fwd_logits, bwd_logits), (fwd_tgt, bwd_tgt), bow_logits, mu_p, log_var_p, mu_r, log_var_r

    # normal_vector ~ N(0,1): [seq_len, batch_size, latent_dim]
    # keyword_ids: [batch_size]
    def generate(self, keyword_ids, normal_vector, approach, gen_options):
        assert not self.training
        assert approach in ("beam", "greedy")
        return getattr(self, "_gen_{}".format(approach))(keyword_ids, normal_vector, **gen_options)

    # input: [seq_len, batch_size], the first token of each sequence should be <SOS>
    # hidden_step: [num_layers, batch_size, hidden_size]
    # normal_vector_step ~ N(0,1): [batch_size, latent_dim]
    def _gen_forward_step(self, input, hidden_step, normal_vector_step, use_cache=False):
        segment_ids = None
        if self.opts.need_segment_ids:
            segment_ids = self.embedding.get_segment_ids(input, use_cache=use_cache, restrict=False)
        embedded = self.embedding(input, 0, segment_ids=segment_ids, segment_emb_restrict=False)
        z = self.latent_module.forward_gen_path(hidden_step[-1], normal_vector_step)
        hidden_step = self.fwd_decoder(torch.cat([embedded[-1], z], dim=-1), hidden_step)
        out_proj_inp = hidden_step[-1]
        if self.opts.latent_out_attach:
            out_proj_inp = torch.cat([hidden_step[-1], z], dim=-1)
        logits_step = self.fwd_out_proj(out_proj_inp)
        if self.opts.fwd_use_logits_mask:
            logits_step = self.fwd_logits_mask_layer(
                logits_step,
                use_cache=use_cache,
                decoder_input=input,
                sample_n_to_check=1,
                only_last_step=True)
        return logits_step, hidden_step

    # keyword_ids: [batch_size]
    # normal_vector: [seq_len, batch_size, latent_dim]
    def _gen_greedy(self, keyword_ids, normal_vector, **kwargs):
        batch_size = normal_vector.size(1)
        device = normal_vector.device
        max_seq_len = self.opts.gen_max_seq_len

        keyword_embs = self.embedding.forward_word_emb(keyword_ids)
        hidden = self._init_states(keyword_embs, "fwd")
        input = torch.full((1, batch_size), self.SOS_token, dtype=torch.long, device=device)
        lens = torch.full((batch_size,), max_seq_len, dtype=torch.long, device=device)

        output_steps = []
        for step in range(max_seq_len):
            logits_step, hidden = self._gen_forward_step(input, hidden, normal_vector[step], use_cache=True)
            out_step = logits_step.argmax(dim=-1, keepdim=False)
            output_steps.append(out_step)
            lens[(out_step == self.EOS_token) & (lens == max_seq_len)] = step + 1
            if step == max_seq_len - 1 or (lens < max_seq_len).all():
                break
            input = torch.cat([input, out_step.unsqueeze(0)], dim=0)
        output = torch.stack(output_steps, dim=0)

        if self.opts.need_segment_ids:
            self.embedding.clear_segment_emb_cache()
        if self.opts.fwd_need_remain_syllables:
            self.fwd_logits_mask_layer.clear_cache()

        return output

    def _gen_beam(self, keyword_ids, normal_vector, **kwargs):
        device = normal_vector.device
        _, batch_size, latent_dim = normal_vector.size()
        max_seq_len = self.opts.gen_max_seq_len
        beam_width = kwargs["beam_width"]
        length_norm = kwargs["length_norm"]
        n_best = kwargs["n_best"]

        input = torch.full((1, batch_size), fill_value=self.SOS_token, dtype=torch.long, device=device)
        keyword_embs = self.embedding.forward_word_emb(keyword_ids)
        hidden = self._init_states(keyword_embs, "fwd")
        output_step = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        back_pointers = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        batch_beams = [Beam(beam_width, length_norm, self.EOS_token, n_best) for _ in range(batch_size)]

        # first step
        logits_step, hidden = self._gen_forward_step(input, hidden, normal_vector[0], use_cache=False)
        step_batch_beams(batch_beams, logits_step, output_step, func="init_beams")

        # remain steps
        input = input.repeat_interleave(beam_width, dim=1)
        normal_vector = normal_vector.repeat_interleave(beam_width, dim=1)
        input = torch.cat([input, output_step.unsqueeze(0)], dim=0)
        hidden = hidden.repeat_interleave(beam_width, dim=1)
        for step in range(1, max_seq_len):
            logits_step, hidden = self._gen_forward_step(input, hidden, normal_vector[step], use_cache=False)
            logits_step = logits_step.view(batch_size, beam_width, -1)
            step_batch_beams(batch_beams, logits_step, output_step, back_pointers, func="update_beams")
            if all(b.done for b in batch_beams):
                break
            input = input.index_select(dim=1, index=back_pointers)
            input = torch.cat([input, output_step.unsqueeze(0)], dim=0)
            hidden = hidden.index_select(dim=1, index=back_pointers)

        output = list(chain(*(beam.get_best_results()[0] for beam in batch_beams)))
        output = bidirectional_padding(output, self.PAD_token, 0, device=device)[0]

        return output
