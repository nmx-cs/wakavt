# -*- coding: utf-8 -*-

from modules.embedding import Embeddings
from modules.linear import SharedLinearNoBias
from modules.common_layers import TransformerLayer, LatentLayer, LogitsMaskLayer, TanhFusion
from modules.beam import Beam, step_batch_beams
from utils import generate_triangle_attention_mask, generate_segment_attention_mask, generate_attention_mask_by_set_partition
from utils import xavier_uniform_fan_in_, bidirectional_padding

import torch
from torch import nn
from itertools import chain


def build_model(opts, word2syllable, pretrained_emb_weights):
    if opts.model == "VT":
        return SequentialTransformerCVAE(opts, word2syllable, pretrained_emb_weights)
    elif opts.model == "T":
        return TransformerCVAE(opts, word2syllable, pretrained_emb_weights)
    elif opts.model == "L":
        return TransformerCLM(opts, word2syllable, pretrained_emb_weights)
    else:
        raise ValueError("unknown model type: {}".format(opts.model))


def _build_embeddings(opts, word2syllable, pretrained_emb_weights, use_pos_emb=False):
    nonfrozen_tokens = [getattr(opts, token) for token in opts.name2stoken.keys() if token != "PAD_token"]
    return Embeddings(
        word2syllable,
        pretrained_embs=pretrained_emb_weights,
        use_syllable_emb=opts.use_syllable_emb,
        use_pos_emb=use_pos_emb,
        use_segment_emb=opts.use_segment_emb,
        embs_finetune_option=opts.embeddings_to_finetune,
        PAD_token=opts.PAD_token,
        SEP_token=opts.SEP_token,
        dropout=opts.dropout,
        vocab_size=opts.vocab_size,
        embedding_dim=opts.embedding_dim,
        output_dim=opts.d_model,
        nonfrozen_tokens=nonfrozen_tokens,
        pos_emb_type=opts.pos_emb_type,
        max_seq_len=opts.emb_max_seq_len,
        pos_emb_align_pos=0,
        segment_emb_relative=opts.segment_emb_relative,
        sep_as_new_segment=opts.sep_as_new_segment,
        num_segments=5,
        initialization_method=opts.embedding_initialization)


def _build_stacked_transformer_layers(opts, num_layers, num_self_attn_per_layer, prebuilt_layers=None):
    prebuilt_SAs = [None] * num_layers
    prebuilt_LNs = [None] * num_layers
    prebuilt_MGs = [None] * num_layers
    prebuilt_FFs = [None] * num_layers
    prebuilt_FFLNs = [None] * num_layers
    if prebuilt_layers is not None:
        assert isinstance(prebuilt_layers, nn.ModuleList) and len(prebuilt_layers) == num_layers
        for i, layer in enumerate(prebuilt_layers):
            assert isinstance(layer, TransformerLayer)
            prebuilt_SAs[i] = layer.get_module_by_name("SA") if opts.share_SA_before_latent else None
            prebuilt_LNs[i] = layer.get_module_by_name("LN") if opts.share_LN_before_latent else None
            prebuilt_MGs[i] = layer.get_module_by_name("MG") if opts.share_MG_before_latent else None
            prebuilt_FFs[i] = layer.get_module_by_name("FF") if opts.share_FF_before_latent else None
            prebuilt_FFLNs[i] = layer.get_module_by_name("FFLN") if opts.share_FFLN_before_latent else None
    return nn.ModuleList([TransformerLayer(
            opts.d_model,
            opts.nhead,
            opts.dim_feedforward,
            opts.dropout,
            num_self_attn_per_layer,
            opts.self_attn_out_merge,
            opts.self_attn_share_params,
            use_gelu=opts.use_gelu,
            prebuilt_SA=prebuilt_SAs[i],
            prebuilt_LN=prebuilt_LNs[i],
            prebuilt_MG=prebuilt_MGs[i],
            prebuilt_FF=prebuilt_FFs[i],
            prebuilt_FFLN=prebuilt_FFLNs[i])
            for i in range(num_layers)])


def _build_output_layers(opts, proj_weight=None, emb_weight=None):
    hidden2emb = nn.Identity()
    dim_inp = opts.d_model
    if opts.latent_out_attach:
        assert not opts.weight_typing
        dim_inp += opts.latent_dim
    if opts.weight_typing:
        if opts.d_model != opts.embedding_dim["word"]:
            assert proj_weight is not None
            hidden2emb = SharedLinearNoBias(proj_weight, in_feature_dim=0)
        assert emb_weight is not None
        out_proj = SharedLinearNoBias(emb_weight, in_feature_dim=1, parted_block=[opts.PAD_token, slice(None, None)])
    else:
        out_proj = nn.Linear(dim_inp, opts.vocab_size, bias=False)
    return hidden2emb, out_proj


class ModelBase(nn.Module):
    def __init__(self):
        super(ModelBase, self).__init__()

    def _reset_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, src, keyword_ids, segment_ids=None, remain_syllables=None, mode="train"):
        raise NotImplementedError

    def generate(self, *args, **kwargs):
        raise NotImplementedError

    # tgt: [seq_len, batch_size * sample_n_to_check]
    # keyword_ids: [seq_len, batch_size]
    def get_real_tgt(self, tgt, keyword_ids, sample_n_to_check):
        if keyword_ids is not None:
            if tgt.size(1) != keyword_ids.size(0):
                assert tgt.size(1) == keyword_ids.size(0) * sample_n_to_check
                keyword_ids = keyword_ids.repeat(sample_n_to_check)
            if "ktoken" in self.opts.keyword_approaches:
                tgt = tgt.masked_fill(tgt == keyword_ids.unsqueeze(0), self.opts.KEYWORD_token)
        return tgt

    def get_bow_tgt(self, tgt):
        if not self.opts.is_variational_autoregressive:
            return tgt
        seq_len, batch_size = tgt.size()
        mask = generate_triangle_attention_mask(seq_len, device=tgt.device, dtype=torch.bool)
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

    def _get_attn_masks(self, segment_ids, seq_len, device, triangle, revise_for_khead, revise_for_cls):
        assert not (revise_for_khead and revise_for_cls)
        masks = []
        if "segment" in self.opts.self_attn_levels:
            masks.append(generate_segment_attention_mask(segment_ids, triangle=triangle))
        if "sent" in self.opts.self_attn_levels:
            masks.append(generate_attention_mask_by_set_partition(segment_ids, [{0, 1, 2}, {3, 4}], triangle=triangle))
        if "global" in self.opts.self_attn_levels:
            if triangle:
                masks.append(generate_triangle_attention_mask(seq_len, device=device))
            else:
                masks.append(torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device))
        if revise_for_khead:
            masks = [self.revise_attn_mask_for_prepended(m, prepended="khead") for m in masks]
        if revise_for_cls:
            masks = [self.revise_attn_mask_for_prepended(m, prepended="cls") for m in masks]
        return masks

    def _get_padding_mask(self, input, revise_for_khead, revise_for_cls):
        assert input.dtype == torch.long
        padding_mask = input == self.opts.PAD_token
        if revise_for_khead or revise_for_cls:
            zeros = torch.zeros(padding_mask.size(1), dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([zeros.unsqueeze(0), padding_mask], dim=0)
        return padding_mask.t()

    def _forward_layers(self, module_name, input, attn_masks, key_padding_mask):
        layers = getattr(self, module_name)
        output = input
        for i in range(len(layers)):
            output = layers[i](output, attn_masks, key_padding_mask)
        return output

    def _forward_layers_hierarchical(self, module_name, input, attn_masks, key_padding_mask):
        assert isinstance(attn_masks, list) or isinstance(attn_masks, tuple)
        layers = getattr(self, module_name)
        assert len(layers) == len(attn_masks)
        output = input
        for i in range(len(layers)):
            output = layers[i](output, attn_masks[i], key_padding_mask)
        return output

    @staticmethod
    def _expand_attn_masks(attn_masks, sample_n):
        expanded = []
        for mask in attn_masks:
            if mask.dim() == 2:
                expanded.append(mask)
            elif sample_n > 1:
                expanded.append(mask.repeat(sample_n, 1, 1))
        return expanded

    @staticmethod
    def _expand_padding_mask(padding_mask, sample_n):
        if sample_n > 1:
            padding_mask = padding_mask.repeat(sample_n, 1)
        return padding_mask

    @staticmethod
    def expand_tgt(tgt, sample_n):
        if sample_n > 1:
            tgt = tgt.repeat(1, sample_n)
        return tgt

    @staticmethod
    def revise_attn_mask_for_prepended(attn_mask, prepended=None):
        assert prepended in ("khead", "cls", None)
        ndim = attn_mask.dim()
        if ndim == 2:
            attn_mask = attn_mask.unsqueeze(0)
        fill_value = 0
        if prepended == "khead":
            fill_value = 1 if attn_mask.dtype == torch.bool else float("-inf")
        new_mask = torch.full(
            (attn_mask.size(0), attn_mask.size(1) + 1, attn_mask.size(2) + 1),
            fill_value,
            dtype=attn_mask.dtype,
            device=attn_mask.device)
        if prepended == "khead":
            new_mask[:, :, 0] = 0
        new_mask[:, 1:, 1:] = attn_mask
        return new_mask[0] if ndim == 2 else new_mask


class VTLatentModule(nn.Module):
    def __init__(self, hidden_size, latent_dim, use_tanh):
        super(VTLatentModule, self).__init__()

        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.use_tanh = use_tanh

        self.prior_net = LatentLayer(hidden_size, latent_dim, use_tanh)
        self.recognition_net = LatentLayer(hidden_size, latent_dim, use_tanh)
        self.merge = TanhFusion(hidden_size + latent_dim, hidden_size)

    def forward_train_path(self, prior_input, recognition_input, normal_vector=None, sample_n=1):
        assert prior_input.shape == recognition_input.shape
        mu_p, log_var_p, _ = self.prior_net(prior_input, sample_n=sample_n, reparam=False)
        mu_r, log_var_r, latent_vector = self.recognition_net(recognition_input, normal_vector, sample_n=sample_n)
        latent_out = self.merge(None, latent_vector, prior_input.unsqueeze(1).expand(-1, sample_n, -1, -1))
        return mu_p, log_var_p, mu_r, log_var_r, latent_out, latent_vector

    def forward_gen_path(self, prior_input, normal_vector=None):
        *_, latent_vector = self.prior_net(prior_input, normal_vector, sample_n=1)
        latent_out = self.merge(None, latent_vector, prior_input.unsqueeze(1))
        return latent_out, latent_vector


class SequentialTransformerCVAE(ModelBase):
    def __init__(self, opts, word2syllable, pretrained_emb_weights):
        super(SequentialTransformerCVAE, self).__init__()

        self.opts = opts
        self.PAD_token = opts.PAD_token
        self.SOS_token = opts.SOS_token
        self.EOS_token = opts.EOS_token
        self.UNK_token = opts.UNK_token
        self.SEP_token = opts.SEP_token
        self.KEYWORD_token = opts.KEYWORD_token
        self.keyword_approaches = opts.keyword_approaches

        self.embedding = _build_embeddings(opts, word2syllable, pretrained_emb_weights, use_pos_emb=True)

        self.prior_encoder = _build_stacked_transformer_layers(
            opts,
            opts.num_layers_before_latent,
            opts.num_self_attn_per_layer_before_latent)

        self.recognition_encoder = _build_stacked_transformer_layers(
            opts,
            opts.num_layers_before_latent,
            opts.num_self_attn_per_layer_before_latent,
            prebuilt_layers=self.prior_encoder)

        self.latent_module = VTLatentModule(opts.d_model, opts.latent_dim, opts.latent_use_tanh)
        self.drop = nn.Dropout(opts.dropout)
        self.norm = nn.LayerNorm(opts.d_model)
        if opts.use_bow_loss:
            self.bow_proj_layer = nn.Linear(opts.latent_dim + opts.d_model, opts.vocab_size)

        self.decoder = _build_stacked_transformer_layers(
            opts,
            opts.num_layers - opts.num_layers_before_latent,
            opts.num_self_attn_per_layer_after_latent)

        self.hidden2emb, self.out_proj = _build_output_layers(
            opts, self.embedding.get_word_emb_out_proj_weight(), self.embedding.get_word_emb_weight())

        if opts.use_logits_mask:
            self.logits_mask_layer = LogitsMaskLayer(
                self.embedding.get_word2syllable_buffer(),
                opts.SEP_token, opts.UNK_token, opts.KEYWORD_token)

        self._reset_parameters()

    def _reset_parameters(self):
        if hasattr(self, "bow_proj_layer"):
            xavier_uniform_fan_in_(self.bow_proj_layer.weight)
            nn.init.zeros_(self.bow_proj_layer.bias)
        if not self.opts.weight_typing:
            xavier_uniform_fan_in_(self.out_proj.weight)

    # src (without sos): [seq_len, batch_size]
    # keyword_ids: [batch_size]
    def forward(self, src, keyword_ids, segment_ids=None, remain_syllables=None, mode="train"):
        assert mode in ("train", "valid", "test")
        if mode != "train":
            assert not self.training
        else:
            assert self.training

        sos = torch.full((src.size(1),), fill_value=self.SOS_token, dtype=torch.long, device=src.device)
        input = torch.cat([sos.unsqueeze(0), src[:-1]], dim=0)
        if self.opts.need_segment_ids:
            if segment_ids is not None:
                segment_ids = torch.cat([torch.zeros_like(segment_ids[:1]), segment_ids[:-1]], dim=0)
            else:
                segment_ids = self.embedding.get_segment_ids(input)
        padding_mask = self._get_padding_mask(input,
                                              revise_for_khead="khead" in self.keyword_approaches,
                                              revise_for_cls=False)
        attn_masks_p = self._get_attn_masks(segment_ids, input.size(0), input.device,
                                            triangle=True,
                                            revise_for_khead="khead" in self.keyword_approaches,
                                            revise_for_cls=False)
        attn_masks_r = self._get_attn_masks(segment_ids, input.size(0), input.device,
                                            triangle=False,
                                            revise_for_khead="khead" in self.keyword_approaches,
                                            revise_for_cls=False)
        sample_n = self.opts.train_sample_n if mode == "train" else self.opts.test_sample_n

        embedded = self.embedding(input, (0, 0), segment_ids=segment_ids)
        if "khead" in self.keyword_approaches:
            keyword_embs = self.embedding.forward_word_emb(keyword_ids)
            embedded = torch.cat([keyword_embs.unsqueeze(0), embedded], dim=0)
        if self.opts.hierarchical_before_latent:
            enc_out_p = self._forward_layers_hierarchical("prior_encoder", embedded, attn_masks_p, padding_mask)
            enc_out_r = self._forward_layers_hierarchical("recognition_encoder", embedded, attn_masks_r, padding_mask)
        elif self.opts.hierarchical_model:
            enc_out_p = self._forward_layers("prior_encoder", embedded, attn_masks_p[-1], padding_mask)
            enc_out_r = self._forward_layers("recognition_encoder", embedded, attn_masks_r[-1], padding_mask)
        else:
            enc_out_p = self._forward_layers("prior_encoder", embedded, attn_masks_p, padding_mask)
            enc_out_r = self._forward_layers("recognition_encoder", embedded, attn_masks_r, padding_mask)

        mu_p, log_var_p, mu_r, log_var_r, latent_out, latent_vector = self.latent_module.forward_train_path(
            enc_out_p, enc_out_r, sample_n=sample_n)
        # mu/log_var: [seq_len, batch_size, latent_dim]; latent_out/latent_vec: [seq_len, sample_n, batch_size, d_model/latent_dim]

        if "khead" in self.keyword_approaches:
            latent_out[0] = 0.
        latent_out = self.norm(enc_out_p.unsqueeze(1) + self.drop(latent_out))
        latent_out = latent_out.view(latent_out.size(0), sample_n * latent_out.size(2), -1)
        if self.opts.pos_emb_type == "sinusoidal":
            pos_encodings = self.embedding.get_unexpanded_pos_encodings(input.size(0)).unsqueeze(1)
            if "khead" in self.keyword_approaches:
                latent_out[1:] = latent_out[1:] + pos_encodings
            else:
                latent_out = latent_out + pos_encodings

        bow_logits = None
        if self.opts.use_bow_loss:
            enc_out_p = enc_out_p.unsqueeze(1).expand(-1, sample_n, -1, -1)
            bow_inp = torch.cat([latent_vector, enc_out_p], dim=-1)
            bow_inp = bow_inp.view(bow_inp.size(0), sample_n * bow_inp.size(2), -1)
            if "khead" in self.keyword_approaches:
                bow_inp = bow_inp[1:]
            bow_logits = self.bow_proj_layer(bow_inp)
            expand_dim = bow_logits.size(0) if self.opts.bow_window is None else self.opts.bow_window
            bow_logits = bow_logits.unsqueeze(0).expand(expand_dim, -1, -1, -1)

        if sample_n > 1:
            attn_masks_p = self._expand_attn_masks(attn_masks_p, sample_n)
            padding_mask = self._expand_padding_mask(padding_mask, sample_n)

        if self.opts.hierarchical_after_latent:
            dec_out = self._forward_layers_hierarchical("decoder", latent_out, attn_masks_p, padding_mask)
        elif self.opts.hierarchical_model:
            dec_out = self._forward_layers("decoder", latent_out, attn_masks_p[-1], padding_mask)
        else:
            dec_out = self._forward_layers("decoder", latent_out, attn_masks_p, padding_mask)
        if "khead" in self.keyword_approaches:
            dec_out = dec_out[1:]
            latent_vector = latent_vector[1:]

        all_hidden = dec_out
        if self.opts.latent_out_attach:
            latent_vector = latent_vector.view(latent_vector.size(0), sample_n * latent_vector.size(2), -1)
            all_hidden = torch.cat([all_hidden, latent_vector], dim=-1)
        logits = self.out_proj(self.hidden2emb(all_hidden))
        if self.opts.use_logits_mask:
            logits = self.logits_mask_layer(
                logits,
                remain_syllables=remain_syllables,
                decoder_input=input,
                solve_ktoken="ktoken" in self.keyword_approaches,
                keyword_ids=keyword_ids,
                sample_n_to_check=sample_n)

        if "khead" in self.keyword_approaches:
            mu_p = mu_p[1:]
            log_var_p = log_var_p[1:]
            mu_r = mu_r[1:]
            log_var_r = log_var_r[1:]

        return logits, mu_p, log_var_p, mu_r, log_var_r, bow_logits

    # normal_vector ~ N(0,1): [seq_len, batch_size, latent_dim]
    # keyword_ids: [batch_size]
    def generate(self, keyword_ids, normal_vector, approach, gen_options):
        assert not self.training
        assert approach in ("beam", "greedy")
        return getattr(self, "_gen_{}".format(approach))(keyword_ids, normal_vector, **gen_options)

    # input: [seq_len, batch_size], the first token of each sequence should be <SOS>
    # keyword_ids: [batch_size]
    # normal_vector ~ N(0,1): [seq_len, batch_size, latent_dim]
    def _gen_forward_step(self, input, keyword_ids, normal_vector, use_cache=False):
        assert input.shape == normal_vector.shape[:-1]

        segment_ids = None
        if self.opts.need_segment_ids:
            segment_ids = self.embedding.get_segment_ids(input, use_cache=use_cache, restrict=False)
        padding_mask = self._get_padding_mask(input,
                                              revise_for_khead="khead" in self.keyword_approaches,
                                              revise_for_cls=False)
        attn_masks = self._get_attn_masks(segment_ids, input.size(0), input.device,
                                          triangle=True,
                                          revise_for_khead="khead" in self.keyword_approaches,
                                          revise_for_cls=False)
        slice_t = slice(1, None) if "khead" in self.keyword_approaches else slice(None, None)

        embedded = self.embedding(input, (0, 0), segment_ids=segment_ids, segment_emb_restrict=False)
        if "khead" in self.keyword_approaches:
            keyword_embs = self.embedding.forward_word_emb(keyword_ids)
            embedded = torch.cat([keyword_embs.unsqueeze(0), embedded], dim=0)
        if self.opts.hierarchical_before_latent:
            enc_out = self._forward_layers_hierarchical("prior_encoder", embedded, attn_masks, padding_mask)
        elif self.opts.hierarchical_model:
            enc_out = self._forward_layers("prior_encoder", embedded, attn_masks[-1], padding_mask)
        else:
            enc_out = self._forward_layers("prior_encoder", embedded, attn_masks, padding_mask)

        latent_out, latent_vector = self.latent_module.forward_gen_path(enc_out[slice_t], normal_vector)
        if "khead" in self.keyword_approaches:
            latent_out = torch.cat([torch.zeros_like(latent_out[:1]), latent_out], dim=0)
        latent_out = self.norm(enc_out + self.drop(latent_out.squeeze(1)))
        if self.opts.pos_emb_type == "sinusoidal":
            pos_encodings = self.embedding.get_unexpanded_pos_encodings(input.size(0)).unsqueeze(1)
            latent_out[slice_t] += pos_encodings

        if self.opts.hierarchical_after_latent:
            dec_out = self._forward_layers_hierarchical("decoder", latent_out, attn_masks, padding_mask)
        elif self.opts.hierarchical_model:
            dec_out = self._forward_layers("decoder", latent_out, attn_masks[-1], padding_mask)
        else:
            dec_out = self._forward_layers("decoder", latent_out, attn_masks, padding_mask)
        dec_out = dec_out[slice_t]

        all_hidden = dec_out
        if self.opts.latent_out_attach:
            all_hidden = torch.cat([all_hidden, latent_vector.squeeze(1)], dim=-1)
        logits = self.out_proj(self.hidden2emb(all_hidden))
        if self.opts.use_logits_mask:
            logits = self.logits_mask_layer(
                logits,
                use_cache=use_cache,
                decoder_input=input,
                solve_ktoken="ktoken" in self.keyword_approaches,
                keyword_ids=keyword_ids,
                sample_n_to_check=1)

        return logits

    def _gen_greedy(self, keyword_ids, normal_vector, **kwargs):
        batch_size = normal_vector.size(1)
        device = normal_vector.device
        max_seq_len = self.opts.gen_max_seq_len

        input = torch.full((1, batch_size), self.SOS_token, dtype=torch.long, device=device)
        lens = torch.full((batch_size,), max_seq_len, dtype=torch.long, device=device)
        output_steps = []
        for step in range(max_seq_len):
            logits = self._gen_forward_step(input, keyword_ids, normal_vector[:input.size(0)], use_cache=True)
            out_step = logits[-1].argmax(dim=-1, keepdim=False)
            output_steps.append(out_step.clone())
            lens[(out_step == self.EOS_token) & (lens == max_seq_len)] = logits.size(0)
            if step == max_seq_len - 1 or (lens < max_seq_len).all():
                break
            if "ktoken" in self.keyword_approaches:
                mask = out_step == self.KEYWORD_token
                out_step[mask] = keyword_ids[mask]
            input = torch.cat([input, out_step.unsqueeze(0)], dim=0)
        output = torch.stack(output_steps, dim=0)

        if self.opts.need_segment_ids:
            self.embedding.clear_segment_emb_cache()
        if self.opts.need_remain_syllables:
            self.logits_mask_layer.clear_cache()

        return output

    def _gen_beam(self, keyword_ids, normal_vector, **kwargs):
        device = normal_vector.device
        _, batch_size, latent_dim = normal_vector.size()
        max_seq_len = self.opts.gen_max_seq_len
        beam_width = kwargs["beam_width"]
        length_norm = kwargs["length_norm"]
        n_best = kwargs["n_best"]

        input = torch.full((1, batch_size), fill_value=self.SOS_token, dtype=torch.long, device=device)
        output_step = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        back_pointers = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        batch_beams = [Beam(beam_width, length_norm, self.EOS_token, n_best) for _ in range(batch_size)]

        # first step
        logits_step = self._gen_forward_step(input, keyword_ids, normal_vector[:1], use_cache=False)[-1]
        step_batch_beams(batch_beams, logits_step, output_step, func="init_beams")
        if keyword_ids is not None:
            keyword_ids = keyword_ids.repeat_interleave(beam_width, dim=0)
        if "ktoken" in self.keyword_approaches:
            mask = output_step == self.KEYWORD_token
            output_step[mask] = keyword_ids[mask]

        # remain steps
        input = input.repeat_interleave(beam_width, dim=1)
        normal_vector = normal_vector.repeat_interleave(beam_width, dim=1)
        input = torch.cat([input, output_step.unsqueeze(0)], dim=0)
        for _ in range(1, max_seq_len):
            logits = self._gen_forward_step(input, keyword_ids, normal_vector[:input.size(0)], use_cache=False)
            logits_step = logits[-1].view(batch_size, beam_width, -1)
            step_batch_beams(batch_beams, logits_step, output_step, back_pointers, func="update_beams")
            if all(b.done for b in batch_beams):
                break
            if "ktoken" in self.keyword_approaches:
                mask = output_step == self.KEYWORD_token
                output_step[mask] = keyword_ids[mask]
            input = input.index_select(dim=1, index=back_pointers)
            input = torch.cat([input, output_step.unsqueeze(0)], dim=0)

        output = list(chain(*(beam.get_best_results()[0] for beam in batch_beams)))
        output = bidirectional_padding(output, self.PAD_token, 0, device=device)[0]

        return output


class TLatentModule(nn.Module):
    def __init__(self, hidden_size, embedding_dim, latent_dim, use_tanh, conditioned_by_keywords):
        super(TLatentModule, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.use_tanh = use_tanh
        self.conditioned_by_keywords = conditioned_by_keywords

        prior_input_dim = 0
        if conditioned_by_keywords:
            prior_input_dim += embedding_dim
        recog_input_dim = prior_input_dim + hidden_size
        self.prior_net = LatentLayer(prior_input_dim, latent_dim, use_tanh)
        self.recognition_net = LatentLayer(recog_input_dim, latent_dim, use_tanh)
        self.latent2emb = nn.Identity()
        if embedding_dim != latent_dim:
            self.latent2emb = nn.Linear(latent_dim, embedding_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.embedding_dim != self.latent_dim:
            xavier_uniform_fan_in_(self.latent2emb.weight)
            nn.init.zeros_(self.latent2emb.bias)

    # keyword_embs: [batch_size, embedding_dim]
    # enc_hidden: [batch_size, hidden_size]
    # normal_vector: [sample_n, batch_size, latent_dim]
    def forward_train_path(self, enc_hidden, keyword_embs=None, normal_vector=None, sample_n=1):
        mu_p, log_var_p, _ = self.prior_net(keyword_embs, sample_n=sample_n, reparam=False,
                                            head_dims=[], batch_size=enc_hidden.size(0),
                                            dtype=enc_hidden.dtype, device=enc_hidden.device)
        if self.conditioned_by_keywords:
            assert keyword_embs is not None
            recog_inp = torch.cat([keyword_embs, enc_hidden], dim=-1)
        else:
            recog_inp = enc_hidden
        mu_r, log_var_r, latent_vector = self.recognition_net(recog_inp, normal_vector, sample_n=sample_n)
        latent_out = self.latent2emb(latent_vector)
        return mu_p, log_var_p, mu_r, log_var_r, latent_vector, latent_out

    def forward_gen_path(self, keyword_embs=None, normal_vector=None, **kwargs):
        *_, latent_vector = self.prior_net(keyword_embs, normal_vector, sample_n=1, **kwargs)
        return latent_vector, self.latent2emb(latent_vector)


class TransformerCVAE(ModelBase):
    def __init__(self, opts, word2syllable, pretrained_emb_weights):
        super(TransformerCVAE, self).__init__()

        self.opts = opts
        self.PAD_token = opts.PAD_token
        self.SOS_token = opts.SOS_token
        self.EOS_token = opts.EOS_token
        self.UNK_token = opts.UNK_token
        self.SEP_token = opts.SEP_token
        self.CLS_token = opts.CLS_token
        self.KEYWORD_token = opts.KEYWORD_token
        self.keyword_approaches = opts.keyword_approaches

        self.embedding = _build_embeddings(opts, word2syllable, pretrained_emb_weights, use_pos_emb=True)

        self.encoder = _build_stacked_transformer_layers(
            opts,
            opts.num_layers_before_latent,
            opts.num_self_attn_per_layer_before_latent)

        self.latent_module = TLatentModule(
            opts.d_model, opts.d_model, opts.latent_dim, opts.latent_use_tanh, "klatent" in opts.keyword_approaches)

        if opts.use_bow_loss:
            bow_inp_dim = opts.latent_dim
            if "klatent" in opts.keyword_approaches:
                bow_inp_dim += opts.d_model
            self.bow_proj_layer = nn.Linear(bow_inp_dim, opts.vocab_size)

        self.decoder = _build_stacked_transformer_layers(
            opts,
            opts.num_layers - opts.num_layers_before_latent,
            opts.num_self_attn_per_layer_after_latent)

        self.hidden2emb, self.out_proj = _build_output_layers(
            opts, self.embedding.get_word_emb_out_proj_weight(), self.embedding.get_word_emb_weight())

        if opts.use_logits_mask:
            self.logits_mask_layer = LogitsMaskLayer(
                self.embedding.get_word2syllable_buffer(),
                opts.SEP_token, opts.UNK_token, opts.KEYWORD_token)

        self._reset_parameters()

    def _reset_parameters(self):
        if hasattr(self, "bow_proj_layer"):
            xavier_uniform_fan_in_(self.bow_proj_layer.weight)
            nn.init.zeros_(self.bow_proj_layer.bias)
        if not self.opts.weight_typing:
            xavier_uniform_fan_in_(self.out_proj.weight)

    # src(without sos): [seq_len, batch_size]
    # keyword_ids: [batch_size]
    def forward(self, src, keyword_ids, segment_ids=None, remain_syllables=None, mode="train"):
        assert mode in ("train", "valid", "test")
        if mode != "train":
            assert not self.training
        else:
            assert self.training

        if self.opts.need_segment_ids and segment_ids is None:
            segment_ids = self.embedding.get_segment_ids(src)
        padding_mask = self._get_padding_mask(src, revise_for_khead=False, revise_for_cls=True)
        attn_masks = self._get_attn_masks(segment_ids, src.size(0), src.device,
                                          triangle=False,
                                          revise_for_khead=False,
                                          revise_for_cls=True)
        embedded = self.embedding(src, (0, 0), segment_ids=segment_ids)
        cls = torch.full((src.size(1),), fill_value=self.CLS_token, dtype=torch.long, device=src.device)
        cls_embs = self.embedding.forward_word_emb(cls)
        embedded = torch.cat([cls_embs.unsqueeze(0), embedded], dim=0)
        if self.opts.hierarchical_before_latent:
            enc_out = self._forward_layers_hierarchical("encoder", embedded, attn_masks, padding_mask)
        elif self.opts.hierarchical_model:
            enc_out = self._forward_layers("encoder", embedded, attn_masks[-1], padding_mask)
        else:
            enc_out = self._forward_layers("encoder", embedded, attn_masks, padding_mask)
        enc_hidden = enc_out[0]

        sample_n = self.opts.train_sample_n if mode == "train" else self.opts.test_sample_n
        keyword_embs = None
        if keyword_ids is not None:
            keyword_embs = self.embedding.forward_word_emb(keyword_ids)
        mu_p, log_var_p, mu_r, log_var_r, latent_vector, latent_out = self.latent_module.forward_train_path(
            enc_hidden, keyword_embs, sample_n=sample_n)
        # mu/log_var: [batch_size, latent_dim]; latent_out/latent_vec: [sample_n, batch_size, embedding_dim/latent_dim]
        bow_logits = None
        if self.opts.use_bow_loss:
            if keyword_embs is not None:
                keyword_embs_expanded = keyword_embs.unsqueeze(0).expand(sample_n, -1, -1)
                bow_inp = torch.cat([latent_vector, keyword_embs_expanded], dim=-1)
            else:
                bow_inp = latent_vector
            bow_inp = bow_inp.view(sample_n * bow_inp.size(1), -1)
            bow_logits = self.bow_proj_layer(bow_inp)
            bow_logits = bow_logits.unsqueeze(0).expand(src.size(0), -1, -1)

        sos = torch.full((src.size(1),), fill_value=self.SOS_token, dtype=torch.long, device=src.device)
        dec_input = torch.cat([sos.unsqueeze(0), src[:-1]], dim=0)
        if self.opts.need_segment_ids:
            segment_ids = torch.cat([torch.zeros_like(segment_ids[:1]), segment_ids[:-1]], dim=0)
        padding_mask = self._get_padding_mask(dec_input,
                                              revise_for_khead="khead" in self.opts.keyword_approaches,
                                              revise_for_cls=False)
        attn_masks = self._get_attn_masks(segment_ids, dec_input.size(0), dec_input.device,
                                          triangle=True,
                                          revise_for_khead="khead" in self.opts.keyword_approaches,
                                          revise_for_cls=False)
        embedded = self.embedding(dec_input, (0, 0), segment_ids=segment_ids)
        if sample_n > 1:
            padding_mask = self._expand_padding_mask(padding_mask, sample_n)
            attn_masks = self._expand_attn_masks(attn_masks, sample_n)
            embedded = embedded.repeat(1, sample_n, 1)
            if "khead" in self.keyword_approaches:
                keyword_embs = keyword_embs.repeat(sample_n, 1)
        embedded[0] = embedded[0] + latent_out.view(sample_n * latent_out.size(1), latent_out.size(-1))
        if "khead" in self.keyword_approaches:
            embedded = torch.cat([keyword_embs.unsqueeze(0), embedded], dim=0)
        if self.opts.hierarchical_after_latent:
            dec_out = self._forward_layers_hierarchical("decoder", embedded, attn_masks, padding_mask)
        elif self.opts.hierarchical_model:
            dec_out = self._forward_layers("decoder", embedded, attn_masks[-1], padding_mask)
        else:
            dec_out = self._forward_layers("decoder", embedded, attn_masks, padding_mask)
        if "khead" in self.keyword_approaches:
            dec_out = dec_out[1:]

        logits = self.out_proj(self.hidden2emb(dec_out))
        if self.opts.use_logits_mask:
            logits = self.logits_mask_layer(
                logits,
                remain_syllables=remain_syllables,
                decoder_input=dec_input,
                solve_ktoken="ktoken" in self.keyword_approaches,
                keyword_ids=keyword_ids,
                sample_n_to_check=sample_n)

        return logits, mu_p, log_var_p, mu_r, log_var_r, bow_logits

    # normal_vector ~ N(0,1): [batch_size, latent_dim]
    # keyword_ids: [batch_size]
    def generate(self, keyword_ids, normal_vector, approach, gen_options):
        assert not self.training
        assert approach in ("beam", "greedy")
        return getattr(self, "_gen_{}".format(approach))(keyword_ids, normal_vector, **gen_options)

    # input: [seq_len, batch_size], the first token of each sequence should be <SOS>
    # keyword_ids: [batch_size]
    # latent_out: [batch_size, embedding_dim]
    def _gen_forward_step(self, input, keyword_ids, keyword_embs, latent_out, use_cache=False):
        segment_ids = None
        if self.opts.need_segment_ids:
            segment_ids = self.embedding.get_segment_ids(input, use_cache=use_cache, restrict=False)
        padding_mask = self._get_padding_mask(input,
                                              revise_for_khead="khead" in self.opts.keyword_approaches,
                                              revise_for_cls=False)
        attn_masks = self._get_attn_masks(segment_ids, input.size(0), input.device,
                                          triangle=True,
                                          revise_for_khead="khead" in self.opts.keyword_approaches,
                                          revise_for_cls=False)
        embedded = self.embedding(input, (0, 0), segment_ids=segment_ids)
        embedded[0] = embedded[0] + latent_out
        if "khead" in self.keyword_approaches:
            embedded = torch.cat([keyword_embs.unsqueeze(0), embedded], dim=0)
        if self.opts.hierarchical_after_latent:
            dec_out = self._forward_layers_hierarchical("decoder", embedded, attn_masks, padding_mask)
        elif self.opts.hierarchical_model:
            dec_out = self._forward_layers("decoder", embedded, attn_masks[-1], padding_mask)
        else:
            dec_out = self._forward_layers("decoder", embedded, attn_masks, padding_mask)
        if "khead" in self.keyword_approaches:
            dec_out = dec_out[1:]

        logits = self.out_proj(self.hidden2emb(dec_out))
        if self.opts.use_logits_mask:
            logits = self.logits_mask_layer(
                logits,
                use_cache=use_cache,
                decoder_input=input,
                solve_ktoken="ktoken" in self.keyword_approaches,
                keyword_ids=keyword_ids,
                sample_n_to_check=1)

        return logits

    def _gen_greedy(self, keyword_ids, normal_vector, **kwargs):
        batch_size = normal_vector.size(0)
        dtype = normal_vector.dtype
        device = normal_vector.device
        max_seq_len = self.opts.gen_max_seq_len

        keyword_embs = None
        if keyword_ids is not None:
            keyword_embs = self.embedding.forward_word_emb(keyword_ids)
        latent_out = self.latent_module.forward_gen_path(keyword_embs, normal_vector,
                                                         head_dims=[], batch_size=batch_size,
                                                         dtype=dtype, device=device)[1].squeeze(0)

        input = torch.full((1, batch_size), self.SOS_token, dtype=torch.long, device=device)
        lens = torch.full((batch_size,), max_seq_len, dtype=torch.long, device=device)
        output_steps = []
        for step in range(max_seq_len):
            logits = self._gen_forward_step(input, keyword_ids, keyword_embs, latent_out, use_cache=True)
            out_step = logits[-1].argmax(dim=-1, keepdim=False)
            output_steps.append(out_step.clone())
            lens[(out_step == self.EOS_token) & (lens == max_seq_len)] = logits.size(0)
            if step == max_seq_len - 1 or (lens < max_seq_len).all():
                break
            if "ktoken" in self.keyword_approaches:
                mask = out_step == self.KEYWORD_token
                out_step[mask] = keyword_ids[mask]
            input = torch.cat([input, out_step.unsqueeze(0)], dim=0)
        output = torch.stack(output_steps, dim=0)

        if self.opts.need_segment_ids:
            self.embedding.clear_segment_emb_cache()
        if self.opts.need_remain_syllables:
            self.logits_mask_layer.clear_cache()

        return output

    def _gen_beam(self, keyword_ids, normal_vector, **kwargs):
        dtype = normal_vector.dtype
        device = normal_vector.device
        batch_size, latent_dim = normal_vector.size()
        max_seq_len = self.opts.gen_max_seq_len
        beam_width = kwargs["beam_width"]
        length_norm = kwargs["length_norm"]
        n_best = kwargs["n_best"]

        keyword_embs = None
        if keyword_ids is not None:
            keyword_embs = self.embedding.forward_word_emb(keyword_ids)
        latent_out = self.latent_module.forward_gen_path(keyword_embs, normal_vector,
                                                         head_dims=[], batch_size=batch_size,
                                                         dtype=dtype, device=device)[1].squeeze(0)

        input = torch.full((1, batch_size), fill_value=self.SOS_token, dtype=torch.long, device=device)
        output_step = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        back_pointers = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        batch_beams = [Beam(beam_width, length_norm, self.EOS_token, n_best) for _ in range(batch_size)]

        # first step
        logits_step = self._gen_forward_step(input, keyword_ids, keyword_embs, latent_out, use_cache=False)[-1]
        step_batch_beams(batch_beams, logits_step, output_step, func="init_beams")
        if keyword_ids is not None:
            keyword_ids = keyword_ids.repeat_interleave(beam_width, dim=0)
        if "ktoken" in self.keyword_approaches:
            mask = output_step == self.KEYWORD_token
            output_step[mask] = keyword_ids[mask]

        # remain steps
        input = input.repeat_interleave(beam_width, dim=1)
        input = torch.cat([input, output_step.unsqueeze(0)], dim=0)
        latent_out = latent_out.repeat_interleave(beam_width, dim=0)
        if keyword_embs is not None:
            keyword_embs = keyword_embs.repeat_interleave(beam_width, dim=0)
        for _ in range(1, max_seq_len):
            logits = self._gen_forward_step(input, keyword_ids, keyword_embs, latent_out, use_cache=False)
            logits_step = logits[-1].view(batch_size, beam_width, -1)
            step_batch_beams(batch_beams, logits_step, output_step, back_pointers, func="update_beams")
            if all(b.done for b in batch_beams):
                break
            if "ktoken" in self.keyword_approaches:
                mask = output_step == self.KEYWORD_token
                output_step[mask] = keyword_ids[mask]
            input = input.index_select(dim=1, index=back_pointers)
            input = torch.cat([input, output_step.unsqueeze(0)], dim=0)

        output = list(chain(*(beam.get_best_results()[0] for beam in batch_beams)))
        output = bidirectional_padding(output, self.PAD_token, 0, device=device)[0]

        return output


class TransformerCLM(ModelBase):
    def __init__(self, opts, word2syllable, pretrained_emb_weights):
        super(TransformerCLM, self).__init__()

        self.opts = opts
        self.PAD_token = opts.PAD_token
        self.SOS_token = opts.SOS_token
        self.EOS_token = opts.EOS_token
        self.UNK_token = opts.UNK_token
        self.SEP_token = opts.SEP_token
        self.KEYWORD_token = opts.KEYWORD_token
        self.keyword_approaches = opts.keyword_approaches

        self.embedding = _build_embeddings(opts, word2syllable, pretrained_emb_weights, use_pos_emb=True)

        self.decoder = _build_stacked_transformer_layers(opts, opts.num_layers, opts.num_self_attn_per_layer)

        self.hidden2emb, self.out_proj = _build_output_layers(
            opts, self.embedding.get_word_emb_out_proj_weight(), self.embedding.get_word_emb_weight())

        if opts.use_logits_mask:
            self.logits_mask_layer = LogitsMaskLayer(
                self.embedding.get_word2syllable_buffer(),
                opts.SEP_token, opts.UNK_token, opts.KEYWORD_token)

        self._reset_parameters()

    def _reset_parameters(self):
        if not self.opts.weight_typing:
            xavier_uniform_fan_in_(self.out_proj.weight)

    def _forward_all(self, input, keyword_ids, segment_ids=None, remain_syllables=None, use_cache=False):
        padding_mask = self._get_padding_mask(input,
                                              revise_for_khead="khead" in self.opts.keyword_approaches,
                                              revise_for_cls=False)
        attn_masks = self._get_attn_masks(segment_ids, input.size(0), input.device,
                                          triangle=True,
                                          revise_for_khead="khead" in self.opts.keyword_approaches,
                                          revise_for_cls=False)
        embedded = self.embedding(input, (0, 0), segment_ids=segment_ids)
        if "khead" in self.keyword_approaches:
            keyword_embs = self.embedding.forward_word_emb(keyword_ids)
            embedded = torch.cat([keyword_embs.unsqueeze(0), embedded], dim=0)
        if self.opts.hierarchical_model:
            dec_out = self._forward_layers_hierarchical("decoder", embedded, attn_masks, padding_mask)
        else:
            dec_out = self._forward_layers("decoder", embedded, attn_masks, padding_mask)
        if "khead" in self.keyword_approaches:
            dec_out = dec_out[1:]
        logits = self.out_proj(self.hidden2emb(dec_out))
        if self.opts.use_logits_mask:
            logits = self.logits_mask_layer(
                logits,
                use_cache=use_cache,
                remain_syllables=remain_syllables,
                decoder_input=input,
                solve_ktoken="ktoken" in self.keyword_approaches,
                keyword_ids=keyword_ids,
                sample_n_to_check=1)
        return logits

    # src(without sos): [seq_len, batch_size]
    # keyword_ids: [batch_size]
    def forward(self, src, keyword_ids, segment_ids=None, remain_syllables=None, mode="train"):
        assert mode in ("train", "valid", "test")
        if mode != "train":
            assert not self.training
        else:
            assert self.training

        sos = torch.full((src.size(1),), fill_value=self.SOS_token, dtype=torch.long, device=src.device)
        input = torch.cat([sos.unsqueeze(0), src[:-1]], dim=0)
        if self.opts.need_segment_ids:
            if segment_ids is not None:
                segment_ids = torch.cat([torch.zeros_like(segment_ids[:1]), segment_ids[:-1]], dim=0)
            else:
                segment_ids = self.embedding.get_segment_ids(input)
        logits = self._forward_all(input, keyword_ids, segment_ids, remain_syllables)

        return (logits,)

    # keyword_ids: [batch_size]
    def generate(self, keyword_ids, approach, gen_options):
        assert not self.training
        assert keyword_ids is not None
        assert approach in ("beam", "greedy")
        return getattr(self, "_gen_{}".format(approach))(keyword_ids, **gen_options)

    # input: [seq_len, batch_size], the first token of each sequence should be <SOS>
    # keyword_ids: [batch_size]
    def _gen_forward_step(self, input, keyword_ids, use_cache=False):
        segment_ids = None
        if self.opts.need_segment_ids:
            segment_ids = self.embedding.get_segment_ids(input, use_cache=use_cache, restrict=False)
        return self._forward_all(input, keyword_ids, segment_ids, use_cache=use_cache)

    def _gen_greedy(self, keyword_ids, **kwargs):
        batch_size = keyword_ids.size(0)
        device = keyword_ids.device
        max_seq_len = self.opts.gen_max_seq_len

        input = torch.full((1, batch_size), self.SOS_token, dtype=torch.long, device=device)
        lens = torch.full((batch_size,), max_seq_len, dtype=torch.long, device=device)
        output_steps = []
        for step in range(max_seq_len):
            logits = self._gen_forward_step(input, keyword_ids, use_cache=True)
            out_step = logits[-1].argmax(dim=-1, keepdim=False)
            output_steps.append(out_step.clone())
            lens[(out_step == self.EOS_token) & (lens == max_seq_len)] = logits.size(0)
            if step == max_seq_len - 1 or (lens < max_seq_len).all():
                break
            if "ktoken" in self.keyword_approaches:
                mask = out_step == self.KEYWORD_token
                out_step[mask] = keyword_ids[mask]
            input = torch.cat([input, out_step.unsqueeze(0)], dim=0)
        output = torch.stack(output_steps, dim=0)

        if self.opts.need_segment_ids:
            self.embedding.clear_segment_emb_cache()
        if self.opts.need_remain_syllables:
            self.logits_mask_layer.clear_cache()

        return output

    def _gen_beam(self, keyword_ids, **kwargs):
        device = keyword_ids.device
        batch_size = keyword_ids.size(0)
        max_seq_len = self.opts.gen_max_seq_len
        beam_width = kwargs["beam_width"]
        length_norm = kwargs["length_norm"]
        n_best = kwargs["n_best"]

        input = torch.full((1, batch_size), fill_value=self.SOS_token, dtype=torch.long, device=device)
        output_step = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        back_pointers = torch.zeros(batch_size * beam_width, dtype=torch.long, device=device)
        batch_beams = [Beam(beam_width, length_norm, self.EOS_token, n_best) for _ in range(batch_size)]

        # first step
        logits_step = self._gen_forward_step(input, keyword_ids, use_cache=False)[-1]
        step_batch_beams(batch_beams, logits_step, output_step, func="init_beams")
        if keyword_ids is not None:
            keyword_ids = keyword_ids.repeat_interleave(beam_width, dim=0)
        if "ktoken" in self.keyword_approaches:
            mask = output_step == self.KEYWORD_token
            output_step[mask] = keyword_ids[mask]

        # remain steps
        input = input.repeat_interleave(beam_width, dim=1)
        input = torch.cat([input, output_step.unsqueeze(0)], dim=0)
        for _ in range(1, max_seq_len):
            logits = self._gen_forward_step(input, keyword_ids, use_cache=False)
            logits_step = logits[-1].view(batch_size, beam_width, -1)
            step_batch_beams(batch_beams, logits_step, output_step, back_pointers, func="update_beams")
            if all(b.done for b in batch_beams):
                break
            if "ktoken" in self.keyword_approaches:
                mask = output_step == self.KEYWORD_token
                output_step[mask] = keyword_ids[mask]
            input = input.index_select(dim=1, index=back_pointers)
            input = torch.cat([input, output_step.unsqueeze(0)], dim=0)

        output = list(chain(*(beam.get_best_results()[0] for beam in batch_beams)))
        output = bidirectional_padding(output, self.PAD_token, 0, device=device)[0]

        return output
