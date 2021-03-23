# -*- coding: utf-8 -*-


import math
import torch


class TextLossBase(object):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    # weight: [seq_len, batch_size]
    @staticmethod
    def text_weights_normalization(weight):
        loss_mask_0_1 = weight.bool().to(weight.dtype)
        nonzero_weights_count = loss_mask_0_1.sum(0, keepdim=False)
        factor = nonzero_weights_count / (weight.sum(0, keepdim=False) + 1e-12)
        normalized_weight = factor.unsqueeze(0) * weight
        return normalized_weight, nonzero_weights_count

    # loss: [seq_len, batch_size]
    # option: batch_mean / steps_mean / batch_wise_mean / batch_wise_sum
    @staticmethod
    def loss_reduction(loss, nonzero_weights_count, option):
        if nonzero_weights_count.dtype != loss.dtype:
            nonzero_weights_count = nonzero_weights_count.to(loss.dtype)
        if option == "batch_mean":
            loss = loss.sum() / loss.size(1)
        elif option == "steps_mean":
            loss = (loss.sum(0, keepdim=False) / (nonzero_weights_count + 1e-12)).mean()
        elif option == "batch_wise_mean":
            loss = loss.sum(0, keepdim=False) / (nonzero_weights_count + 1e-12)
        elif option == "batch_wise_sum":
            loss = loss.sum(0, keepdim=False)
        else:
            raise ValueError("Unknown loss reduction option: {}".format(option))
        return loss


class CrossEntropy(TextLossBase):
    def __init__(self, weight=None, *, PAD_token=None, mask_pad=False):
        super(CrossEntropy, self).__init__()

        assert PAD_token is not None or not mask_pad
        self.PAD_token = PAD_token
        self.weight = weight
        self.mask_pad = mask_pad
        if weight is not None:
            self.weight = weight.detach().clone()
            if mask_pad and weight[PAD_token] != 0.:
                print("WARNING: mask_pad is True but weight[PAD] is not 0, so weight[PAD] will be set to zero.")
                self.weight[PAD_token] = 0.
            self.weight_version_to_check = self.weight._version

    # logits : [seq_len, batch_size, vocab_size]
    # target : [seq_len, batch_size]
    # loss_mask : [seq_len, batch_size]
    # reduction: ["batch_mean", "steps_mean", "batch_wise_mean", "batch_wise_sum"]
    def __call__(self, logits, target, loss_mask=None, use_softmax=True, reduction="batch_mean"):
        seq_len, batch_size, vocab_size = logits.size()
        assert tuple(target.shape) == (seq_len, batch_size)
        if loss_mask is not None:
            assert loss_mask.dtype == torch.bool and loss_mask.shape == target.shape

        if self.weight is None:
            self.weight = torch.ones(vocab_size, dtype=logits.dtype, device=logits.device)
            if self.mask_pad:
                self.weight[self.PAD_token] = 0.
            self.weight_version_to_check = self.weight._version
        else:
            assert self.weight._version == self.weight_version_to_check

        weight = self.weight[target]
        if loss_mask is not None:
            weight.masked_fill_(~loss_mask, 0.)
        log_prob = torch.log_softmax(logits, dim=-1) if use_softmax else torch.log(logits)
        neg_log_prob = torch.gather(log_prob, dim=-1, index=target.unsqueeze(-1)).squeeze(-1).neg()
        normalized_weight, nonzero_weights_count = self.text_weights_normalization(weight)
        loss = normalized_weight * neg_log_prob
        loss = self.loss_reduction(loss, nonzero_weights_count, reduction)

        return loss


class SBOWLoss(CrossEntropy):
    def __init__(self, weight=None, *, PAD_token=None, mask_pad=False):
        super(SBOWLoss, self).__init__(weight, PAD_token=PAD_token, mask_pad=mask_pad)

    # logits : [predict_len, seq_len, batch_size, vocab_size]
    # target : [predict_len, seq_len, batch_size]
    # loss_mask : [seq_len, batch_size]
    # reduction: ["batch_mean", "steps_mean", "batch_wise_mean", "batch_wise_sum"]
    def __call__(self, logits, target, loss_mask=None, use_softmax=True, reduction="batch_mean"):
        predict_len, seq_len, batch_size = target.size()
        assert tuple(logits.size())[:-1] == (predict_len, seq_len, batch_size)
        if loss_mask is not None:
            assert loss_mask.dtype == torch.bool and tuple(loss_mask.shape) == (seq_len, batch_size)

        logits = logits.view(predict_len, seq_len * batch_size, -1)
        target = target.view(predict_len, seq_len * batch_size)
        loss = super(SBOWLoss, self).__call__(logits, target, use_softmax=use_softmax, reduction="batch_wise_mean")
        loss = loss.view(seq_len, batch_size)
        masked_words_count = seq_len
        if loss_mask is not None:
            masked_words_count = loss_mask.to(loss.dtype).sum(0, keepdim=False)
        loss = self.loss_reduction(loss, masked_words_count, reduction)

        return loss


# KL Divergence of two gaussian distributions
class KLDLoss(TextLossBase):
    def __init__(self, *, total_steps, kl_start, kl_stop, kl_n_cycle, kl_ratio, kl_warmup, kl_increase_type, kl_annealing_type):
        super(KLDLoss, self).__init__()

        self.total_steps = total_steps
        self.kl_start = kl_start
        self.kl_stop = kl_stop
        self.kl_n_cycle = kl_n_cycle
        self.kl_ratio = kl_ratio
        self.kl_warmup = kl_warmup
        self.kl_increase_type = kl_increase_type
        self.kl_annealing_type = kl_annealing_type

        self.cur_kl_coeff = None
        self.KL_coeff_list = None
        self.cur_schedule_step = 0

        self.kl_scheduler = self._get_kl_scheduler(self.total_steps, self.cur_schedule_step)

    # infer_mu : [seq_len, batch_size, latent_dim] or [batch_size, latent_dim]
    # infer_log_var : [seq_len, batch_size, latent_dim] or [batch_size, latent_dim]
    # loss_mask : [seq_len, batch_size]
    def __call__(self, infer_mu, infer_log_var, mode, loss_mask=None, prior_mu=None, prior_log_var=None, reduction="batch_mean"):
        assert mode in ("train", "infer")
        assert infer_mu.dim() in (2, 3) and infer_mu.shape == infer_log_var.shape
        if infer_mu.dim() == 2:
            assert reduction == "batch_mean" or reduction == "batch_wise_sum"

        if prior_mu is None:
            prior_mu = torch.zeros_like(infer_mu)
        if prior_log_var is None:
            prior_log_var = torch.zeros_like(infer_log_var)

        kld_loss = torch.exp(infer_log_var - prior_log_var) + \
                   (infer_mu - prior_mu) * (infer_mu - prior_mu) / torch.exp(prior_log_var) + \
                   (prior_log_var - infer_log_var) - 1.
        kld_loss = 0.5 * kld_loss.sum(dim=-1, keepdim=False)

        if infer_mu.dim() == 3:
            if loss_mask is None:
                nonzero_weights_count = kld_loss.size(0)
            else:
                assert loss_mask.shape == kld_loss.shape
                weight = loss_mask
                if weight.dtype != kld_loss.dtype:
                    weight = weight.to(kld_loss.dtype)
                normalized_weight, nonzero_weights_count = self.text_weights_normalization(weight)
                kld_loss = normalized_weight * kld_loss

        if infer_mu.dim() == 3:
            kld_loss = self.loss_reduction(kld_loss, nonzero_weights_count, reduction)
        elif reduction == "batch_mean":
            kld_loss = kld_loss.mean()

        if mode == "train":
            kl_coeff = self.get_next_kl_coefficient()
        elif mode == "infer":
            kl_coeff = 1.
        else:
            raise ValueError("Param mode should be 'train' or 'infer'.")

        return kld_loss, kl_coeff

    def _get_kl_scheduler(self, total_steps, start_step=0):
        # Code adapted from https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb

        def frange_cycle_linear(start, stop, n_steps, n_cycle=4, ratio=0.5, warmup=0.0):
            L = [stop for _ in range(n_steps)]
            period = n_steps / n_cycle
            step = (stop - start) / (period * (ratio - warmup))  # linear schedule

            for c in range(n_cycle):
                v, i = start, 0
                while i < warmup * period and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = v
                    i += 1
                while v <= stop and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = v
                    v += step
                    i += 1
            return L

        def frange_cycle_sigmoid(start, stop, n_steps, n_cycle=4, ratio=0.5, warmup=0.0):
            assert start >= 0 and stop <= 1
            start += 1e-10
            stop -= 1e-10
            start = -math.log((1 - start) / start)
            stop = -math.log((1 - stop) / stop)

            L = [stop for _ in range(n_steps)]
            period = n_steps / n_cycle
            step = (stop - start) / (period * (ratio - warmup))  # step is in [0,1]

            for c in range(n_cycle):
                v, i = start, 0
                while i < warmup * period and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = v
                    i += 1
                while v <= stop and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = 1.0 / (1.0 + math.exp(-v))
                    v += step
                    i += 1
            return L

        def frange_cycle_cosine(start, stop, n_steps, n_cycle=4, ratio=0.5, warmup=0.0):
            assert start >= 0 and stop <= 1
            start = math.acos(1 - 2 * start)
            stop = math.acos(1 - 2 * stop)

            L = [stop for _ in range(n_steps)]
            period = n_steps / n_cycle
            step = (stop - start) / (period * (ratio - warmup))  # step is in [0,1]

            for c in range(n_cycle):
                v, i = start, 0
                while i < warmup * period and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = v
                    i += 1
                while v <= stop and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = 0.5 - .5 * math.cos(v)
                    v += step
                    i += 1
            return L

        def frange_cycle_tanh(start, stop, n_steps, n_cycle=4, ratio=0.5, warmup=0.0):
            assert start >= 0 and stop <= 2
            start += 1e-10
            stop -= 1e-10
            start = math.atanh(start - 1)
            stop = math.atanh(stop - 1)

            L = [stop for _ in range(n_steps)]
            period = n_steps / n_cycle
            step = (stop - start) / (period * (ratio - warmup))  # step is in [0,1]

            for c in range(n_cycle):
                v, i = start, 0
                while i < warmup * period and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = v
                    i += 1
                while v <= stop and int(i + c * period) < n_steps:
                    L[int(i + c * period)] = math.tanh(v) + 1
                    v += step
                    i += 1
            return L

        def frange_cycle(start, stop, n_steps, n_cycle=4, ratio=0.5, warmup=0.0, increase_type="linear"):
            assert start <= stop
            if start == stop:
                return frange_cycle_linear(start, stop, n_steps, 1, 1)
            if increase_type == "linear":
                return frange_cycle_linear(start, stop, n_steps, n_cycle, ratio, warmup)
            if increase_type == "sigmoid":
                return frange_cycle_sigmoid(start, stop, n_steps, n_cycle, ratio, warmup)
            if increase_type == "cosine":
                return frange_cycle_cosine(start, stop, n_steps, n_cycle, ratio, warmup)
            if increase_type == "tanh":
                return frange_cycle_tanh(start, stop, n_steps, n_cycle, ratio, warmup)
            raise ValueError("KL increase_type not exists!")

        if total_steps is None:
            return None

        start = self.kl_start
        stop = self.kl_stop
        n_cycle = self.kl_n_cycle
        ratio = self.kl_ratio
        warmup = self.kl_warmup
        increase_type = self.kl_increase_type

        if self.kl_annealing_type == "constant":
            assert start == stop
            L = frange_cycle_linear(start, stop, total_steps)
        elif self.kl_annealing_type == "monotonic":
            L = frange_cycle(start, stop, total_steps, 1, ratio, warmup, increase_type)
        elif self.kl_annealing_type == "cyclical":
            L = frange_cycle(start, stop, total_steps, n_cycle, ratio, warmup, increase_type)
        else:
            raise ValueError("Param kl_annealing_type should be constant, monotonic or cyclical.")

        if self.cur_kl_coeff is not None:
            assert self.cur_kl_coeff == L[start_step]
        else:
            self.cur_kl_coeff = L[start_step]

        i = start_step
        while i < total_steps:
            yield i, L[i]
            i += 1

    def get_next_kl_coefficient(self):
        i, a = next(self.kl_scheduler)
        self.cur_schedule_step = i
        self.cur_kl_coeff = a
        return a

    def get_cur_state(self):
        state = {
            "cur_kl_coeff": self.cur_kl_coeff,
            "cur_schedule_step": self.cur_schedule_step}
        return state

    def set_cur_state(self, state):
        assert state is not None
        self.cur_kl_coeff = state["cur_kl_coeff"]
        self.cur_schedule_step = state["cur_schedule_step"]
        self.kl_scheduler = self._get_kl_scheduler(self.total_steps, self.cur_schedule_step)
