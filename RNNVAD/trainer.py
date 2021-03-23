# -*- coding: utf-8 -*-

from data_loader import DataLoader, CVADDataLoader
from model import build_model
from modules.loss_functions import CrossEntropy, KLDLoss, SBOWLoss
from utils import cal_amount_of_params, getPPL, timedelta

import torch
from torch import nn, optim
import time
import os
import json
import signal
import math
from collections import defaultdict


class Trainer(object):
    def __init__(self, opts, mode, data_dict, states_path, metrics_path, pretrained_model_path=None):
        assert mode in ("restart", "continue", "test")
        self.opts = opts
        self.mode = mode
        if mode in ("continue", "test"):
            self.train_states = torch.load(states_path, map_location="cpu")
        self.states_path = states_path
        self.metrics_path = metrics_path
        self.do_testing = True if "test" in data_dict.keys() else False

        md_data = {"word2syllable": data_dict.get("word2sylnum", None),
                   "pretrained_emb_weights": data_dict.get("pretrained_embs", None)}
        self.model = build_model(opts, **md_data).to(opts.device)
        cal_amount_of_params(self.model)

        dl_data = dict((k + "_data", data_dict.get(k, None)) for k in ("train", "valid", "test", "gen"))
        dl_data["word2syllable"] = data_dict.get("word2sylnum", None)
        if self.opts.model == "CVAD":
            self.data = CVADDataLoader(
                opts,
                "train" if mode != "test" else "test",
                **dl_data,
                prepare_test=self.do_testing)
        else:
            self.data = DataLoader(
                opts,
                "train" if mode != "test" else "test",
                **dl_data,
                prepare_test=self.do_testing)

        self.optim = getattr(optim, opts.optimizer)(self.model.parameters(), **opts.optim_params)

        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optim,
            step_size=opts.lr_adjust_rate,
            gamma=opts.lr_decay,
            last_epoch=-1)

        self.text_cross_entropy_loss = CrossEntropy(PAD_token=opts.PAD_token, mask_pad=True)

        if opts.is_cvae and opts.use_bow_loss:
            if opts.is_variational_autoregressive:
                self.bow_loss = SBOWLoss(PAD_token=opts.PAD_token, mask_pad=True)
            else:
                self.bow_loss = CrossEntropy(PAD_token=opts.PAD_token, mask_pad=True)

        if opts.is_cvae:
            if mode != "test":
                num_batch = math.ceil(opts.epoch_size / opts.batch_size)
            self.kld_loss = KLDLoss(
                total_steps=opts.num_epochs * num_batch if mode != "test" else None,
                kl_start=opts.kl_start,
                kl_stop=opts.kl_stop,
                kl_n_cycle=opts.kl_n_cycle,
                kl_ratio=opts.kl_ratio,
                kl_warmup=opts.kl_warmup,
                kl_increase_type=opts.kl_increase_type,
                kl_annealing_type=opts.kl_annealing_type)

        if mode == "restart":
            if pretrained_model_path is not None:
                train_states = torch.load(pretrained_model_path, map_location="cpu")
                state_dict = train_states.get("model_state", train_states)
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print("pretrained model loaded from {}:".format(pretrained_model_path))
                print("unexpected keys found in state file: {}".format(unexpected))
                print("sucessfully loaded parameters: {}".format(list(set(state_dict.keys()) - set(unexpected))))
                print("these parameters failed to be loaded so will be randomly initialized: {}".format(missing))
            self.init_states()
            self.init_metrics()
        elif mode in ("continue", "test"):
            metrics_loaded = self.load_states()
            if not metrics_loaded:
                assert mode == "test", "Didn't find records of metrics in state file, so only test mode is allowed."
                self.load_metrics()

    def uninterruptible_save(self, only_metrics=False, states_filename=None):
        dir_path = os.path.dirname(self.states_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        if not only_metrics:
            self.save_states(states_filename)
        self.save_metrics()
        signal.signal(signal.SIGINT, s)

    def save_metrics(self):
        metrics_path = self.metrics_path
        if self.mode == "test":
            metrics_path = os.path.join(
                os.path.dirname(metrics_path),
                "test_{}_".format(self.train_states["train_steps"]) + os.path.basename(metrics_path))
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4, ensure_ascii=False)

    def load_metrics(self):
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

    def load_states(self):
        assert hasattr(self, "train_states")
        self.model.load_state_dict(self.train_states["model_state"], strict=True)
        # Only when generating texts this can be not strict, otherwise incomplete states will be saved.
        self.optim.load_state_dict(self.train_states["optim_state"])
        self.lr_scheduler.load_state_dict(self.train_states["lr_scheduler_state"])
        if self.opts.is_cvae:
            self.kld_loss.set_cur_state(self.train_states["KLD_state"])
            if self.mode != "test":
                _ = self.kld_loss.get_next_kl_coefficient()
        if "metrics" in self.train_states.keys():
            self.metrics = self.train_states["metrics"]
            return True
        return False

    def save_states(self, filename=None):
        dir_path = os.path.dirname(self.states_path)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        self.train_states["model_state"] = self.model.state_dict()
        self.train_states["optim_state"] = self.optim.state_dict()
        self.train_states["lr_scheduler_state"] = self.lr_scheduler.state_dict()
        if self.opts.is_cvae:
            self.train_states["KLD_state"] = self.kld_loss.get_cur_state()
        self.train_states["opts"] = self.opts.dumps_opts()
        self.train_states["metrics"] = self.metrics

        train_steps = self.train_states["train_steps"]
        if filename is None:
            filename = "step_%d.pt" % train_steps
        file_path = os.path.join(dir_path, filename)
        print("Saving checkpoint: step_%d at %s" % (train_steps, file_path))
        torch.save(self.train_states, file_path)

    def learning_rate_decay(self):
        cur_lr = self.lr_scheduler.get_last_lr()[0]
        if cur_lr > self.opts.min_lr:
            self.lr_scheduler.step()
            adjusted_lr = self.lr_scheduler.get_last_lr()[0]
            if cur_lr != adjusted_lr:
                print("\nDecaying learning rate to", adjusted_lr, "\n")

    def _forward(self, data_iter, mode="train"):
        assert mode in ("train", "valid", "test")
        inputs, lens, keywords, keyword_pos, *args = next(data_iter)
        batch_size = keywords.size(0)
        outs = self.model(inputs, keywords, *args, mode=mode)
        assert isinstance(outs, tuple)
        logits, tgt, *vars = outs
        assert isinstance(logits, tuple) and isinstance(tgt, tuple) and len(logits) == len(tgt) <= 2
        fwd_logits, *bwd_logits = logits
        fwd_tgt, *bwd_tgt = tgt
        if self.opts.is_cvae:
            bow_logits, mu_p, log_var_p, mu_r, log_var_r = vars

        fwd_ce = self.text_cross_entropy_loss(fwd_logits, fwd_tgt, reduction=self.opts.loss_reduction)
        fwd_ppl = getPPL(fwd_logits, fwd_tgt, fwd_tgt != self.opts.PAD_token)
        bwd_ce = torch.zeros_like(fwd_ce)
        bwd_ppl = torch.zeros_like(fwd_ppl)
        if len(bwd_logits) > 0 and self.opts.bwd_ce_coeff > 0:
            bwd_ce = self.text_cross_entropy_loss(bwd_logits[0], bwd_tgt[0], reduction=self.opts.loss_reduction)
            bwd_ppl = getPPL(bwd_logits[0], bwd_tgt[0], bwd_tgt[0] != self.opts.PAD_token)

        loss_mask = fwd_tgt != self.opts.PAD_token
        cur_bow_loss = torch.zeros_like(fwd_ce)
        if self.opts.is_cvae and self.opts.use_bow_loss:
            bow_tgt = self.model.get_bow_tgt(fwd_tgt)
            cur_bow_loss = self.bow_loss(bow_logits, bow_tgt, loss_mask=loss_mask, reduction=self.opts.loss_reduction)
        cur_elbo = cur_kld = torch.zeros_like(fwd_ce)
        kld_coeff = 0.
        if self.opts.is_cvae:
            kld_mode = "infer" if mode != "train" else "train"
            cur_kld, kld_coeff = self.kld_loss(mu_r, log_var_r, kld_mode, loss_mask=loss_mask, prior_mu=mu_p,
                                               prior_log_var=log_var_p, reduction=self.opts.loss_reduction)
            cur_elbo = fwd_ce + cur_kld

        return {"cur_fwd_ce": fwd_ce,
                "cur_bwd_ce": bwd_ce,
                "cur_bow_loss": cur_bow_loss,
                "cur_kld": cur_kld,
                "kld_coeff": kld_coeff,
                "cur_elbo": cur_elbo,
                "cur_fwd_ppl": fwd_ppl,
                "cur_bwd_ppl": bwd_ppl,
                "batch_size": batch_size}

    def train(self):
        self.model.train()

        num_epochs = self.opts.num_epochs
        batch_size = self.opts.batch_size
        num_batch = math.ceil(self.opts.epoch_size / batch_size)
        total_steps = num_epochs * num_batch
        start_step = self.train_states["train_steps"]
        remain_steps = total_steps - start_step
        print("total_steps : {}".format(total_steps))

        # data
        train_iter = self.data.train_iterator(start_step % num_batch)

        if start_step == 0:
            print("{}\tStart training......".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        else:
            print("{}\tContinue training......".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

        # time
        step_stime = time.time()

        for _ in range(remain_steps):
            self.train_states["train_steps"] += 1

            var_dict = self._forward(train_iter, mode="train")
            cur_fwd_ce = var_dict["cur_fwd_ce"]
            cur_bwd_ce = var_dict["cur_bwd_ce"]
            cur_kld = var_dict["cur_kld"]
            kld_coeff = var_dict["kld_coeff"]
            cur_bow_loss = var_dict["cur_bow_loss"]

            cur_loss = cur_fwd_ce * self.opts.fwd_ce_coeff + cur_bwd_ce * self.opts.bwd_ce_coeff + \
                       cur_kld * kld_coeff + cur_bow_loss * self.opts.bow_coeff
            if torch.isnan(cur_loss):
                print("ERROR: NAN detected.")
                exit(0)
            self.optim.zero_grad()
            cur_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.max_gradient_norm)
            self.optim.step()
            self.learning_rate_decay()

            cur_loss = cur_fwd_ce * self.opts.fwd_ce_coeff + cur_bwd_ce * self.opts.bwd_ce_coeff + \
                       cur_kld + cur_bow_loss * self.opts.bow_coeff
            var_dict["cur_loss"] = cur_loss

            # update train_states
            for k, v in var_dict.items():
                if k.startswith("cur_"):
                    if (self.train_states["train_steps"] - 1) % self.opts.logging_rate + self.opts.logging_window \
                            >= self.opts.logging_rate:
                        self.train_states[k]["step"] += v.item()
                    if (self.train_states["train_steps"] - 1) % self.opts.validation_rate + self.opts.validation_window \
                            >= self.opts.validation_rate:
                        self.train_states[k]["mean"] += v.item()

            for k, v in var_dict.items():
                if k.startswith("cur_"):
                    self.metrics["trnes_" + k[4:]].append(v.item())

            # logging
            if self.train_states["train_steps"] % self.opts.logging_rate == 0:
                duration = time.time() - step_stime
                tc = self.metrics["train_time"] + self.metrics["valid_time"] + duration
                rte = (total_steps - self.train_states["train_steps"]) * tc / self.train_states["train_steps"]

                print("Step {}  progress rate: {:.3f}  loss: {:.8f}  ppl: {:.6f}  "
                      "speed: {:.0f} samples per second  remaining (approximately): {}".format(
                    self.train_states["train_steps"],
                    self.train_states["train_steps"] / total_steps,
                    self.train_states["cur_loss"]["step"] / self.opts.logging_window,
                    self.train_states["cur_fwd_ppl"]["step"] / self.opts.logging_window,
                    self.opts.logging_rate * batch_size / duration,
                    timedelta(rte)))

                # update metrics
                self.metrics["train_steps"].append(self.train_states["train_steps"])
                self.metrics["train_time"] += duration
                self.metrics["kl_coefficient"].append(kld_coeff)
                for k in var_dict.keys():
                    if k.startswith("cur_"):
                        self.metrics["train_" + k[4:]].append(
                            self.train_states[k]["step"] / self.opts.logging_window)

                # update train_states
                for k in var_dict.keys():
                    if k.startswith("cur_"):
                        self.train_states[k]["step"] = 0.0

                # reset timing
                step_stime = time.time()

            del var_dict

            # valid
            if self.train_states["train_steps"] % self.opts.validation_rate == 0 or \
                    self.train_states["train_steps"] in self.opts.specific_validations:
                step_stime += self.validate()

            # after each epoch
            if self.train_states["train_steps"] % num_batch == 0:
                completed_epochs = self.train_states["train_steps"] // num_batch
                print("-------------------Epoch Completed %d/%d-------------------" %
                      (completed_epochs, num_epochs))

            # save states
            temp_stime = time.time()
            self.maybe_save_checkpoints()
            step_stime += time.time() - temp_stime

        if -1 in self.opts.specific_validations:
            self.validate()
        if -1 in self.opts.specific_checkpoints:
            if self.do_testing:
                self.test()
            self.uninterruptible_save(False)
        else:
            self.uninterruptible_save(True)

        print("{}\tFinished!".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

    def _evaluate(self, data_iter, mode):
        assert mode in ("valid", "test")
        self.model.eval()

        if mode == "test":
            cur_test_metrics = self.metrics["test_metrics"][self.train_states["train_steps"]]

        batch_size = self.opts.batch_size
        if mode == "valid":
            num_batch = math.ceil(self.opts.valid_size / batch_size)
        else:
            num_batch = math.ceil(self.opts.test_size / batch_size)
        records = {"loss": 0.0, "fwd_ce": 0.0, "bwd_ce": 0.0, "kld": 0.0, "elbo": 0.0,
                   "fwd_ppl": 0.0, "bwd_ppl": 0.0, "bow_loss": 0.0}

        # time
        stime = time.time()

        # update metrics
        if mode == "valid":
            valid_interval = False
            self.metrics["valid_steps"].append(self.train_states["train_steps"])
            if self.train_states["train_steps"] % self.opts.validation_rate == 0:
                valid_interval = True
            num_steps_since_last = self.opts.validation_window if valid_interval else self.train_states["train_steps"] % self.opts.validation_rate
            num_steps_since_last = min(num_steps_since_last, self.opts.validation_window)
            if not valid_interval:
                lind, rind = self.train_states["train_steps"] - num_steps_since_last, self.train_states["train_steps"]
                for k in records.keys():
                    self.train_states["cur_" + k]["mean"] = sum(self.metrics["trnes_" + k][lind:rind])

        with torch.no_grad():
            for _ in range(num_batch):
                var_dict = self._forward(data_iter, mode=mode)
                cur_bsz = var_dict["batch_size"]
                var_dict["cur_fwd_ce"] = cur_fwd_ce = var_dict["cur_fwd_ce"] * cur_bsz
                var_dict["cur_bwd_ce"] = cur_bwd_ce = var_dict["cur_bwd_ce"] * cur_bsz
                var_dict["cur_kld"] = cur_kld = var_dict["cur_kld"] * cur_bsz
                var_dict["cur_elbo"] = var_dict["cur_fwd_ce"] + var_dict["cur_kld"]
                var_dict["cur_bow_loss"] = cur_bow_loss = var_dict["cur_bow_loss"] * cur_bsz
                var_dict["cur_fwd_ppl"] *= cur_bsz
                var_dict["cur_bwd_ppl"] *= cur_bsz
                cur_loss = cur_fwd_ce * self.opts.fwd_ce_coeff + cur_bwd_ce * self.opts.bwd_ce_coeff + \
                           cur_kld + cur_bow_loss * self.opts.bow_coeff
                var_dict["cur_loss"] = cur_loss
                var_dict.pop("kld_coeff")

                for k in records.keys():
                    records[k] += var_dict["cur_" + k].item()

        duration = time.time() - stime
        if mode == "valid":
            self.metrics["valid_time"] += duration
        else:
            cur_test_metrics["time"] = duration

        # logging
        if mode == "valid":
            print(
                "Validate:  valid loss: %.8f  train loss: %.8f  valid ppl: %.6f  train ppl: %.6f  speed: %.0f samples "
                "per second" %
                (records["loss"] / self.opts.valid_size,
                 self.train_states["cur_loss"]["mean"] / num_steps_since_last,
                 records["fwd_ppl"] / self.opts.valid_size,
                 self.train_states["cur_fwd_ppl"]["mean"] / num_steps_since_last,
                 self.opts.valid_size / duration))
        else:
            print("Test:  test loss: %.8f  test ppl: %.6f speed: %.0f samples per second" %
                  (records["loss"] / self.opts.test_size,
                   records["fwd_ppl"] / self.opts.test_size,
                   self.opts.test_size / duration))

        # update metrics
        if mode == "valid":
            for k, v in records.items():
                self.metrics["valid_" + k]["valid"].append(v / self.opts.valid_size)
                self.metrics["valid_" + k]["train"].append(self.train_states["cur_" + k]["mean"] / num_steps_since_last)
        else:
            for k, v in records.items():
                cur_test_metrics[k] = v / self.opts.test_size

        if mode == "valid" and valid_interval:
            # update train_states
            for k in var_dict.keys():
                if k.startswith("cur_"):
                    self.train_states[k]["mean"] = 0.0

        self.model.train()

        return duration

    def validate(self):
        if len(self.metrics["valid_steps"]) > 0 and self.train_states["train_steps"] == self.metrics["valid_steps"][-1]:
            return 0
        valid_iter = self.data.valid_iterator()
        return self._evaluate(valid_iter, "valid")

    def test(self):
        test_iter = self.data.test_iterator()
        if test_iter is None:
            return None
        return self._evaluate(test_iter, "test")

    def maybe_save_checkpoints(self):
        if self.train_states["train_steps"] % self.opts.logging_rate == 0 or \
           self.train_states["train_steps"] % self.opts.validation_rate == 0 or \
           self.train_states["train_steps"] in self.opts.specific_validations:
            self.uninterruptible_save(True)

        if self.train_states["train_steps"] % self.opts.checkpoint_rate == 0:
            only_metrics = True
            if self.opts.ckpt_start_step is None:
                self.opts.ckpt_start_step = 0
            if self.opts.ckpt_start_step <= self.train_states["train_steps"] and \
                    (self.opts.ckpt_stop_step is None or self.train_states["train_steps"] <= self.opts.ckpt_stop_step):
                only_metrics = False
            if self.do_testing and not only_metrics:
                self.test()
            self.uninterruptible_save(only_metrics)

        if self.train_states["train_steps"] in self.opts.specific_checkpoints:
            if self.do_testing:
                self.test()
            self.uninterruptible_save(False)

        if "kld_range" in self.opts.ckpt_autosave_approaches:
            kld_min, kld_max, min_change, start_step, do_valid, max_files = self.opts.ckpt_autosave_approaches["kld_range"]
            assert min_change != 0
            cmp_func = lambda x, y: x <= y if y < 0 else x >= y
            if not hasattr(self, "ckpt_autosave_kld_range_num_saved"):
                self.ckpt_autosave_kld_range_num_files = 0
            if not hasattr(self, "ckpt_autosave_kld_range_last_kld"):
                self.ckpt_autosave_kld_range_last_kld = None
            if self.train_states["train_steps"] >= start_step:
                cur_kld = self.metrics["train_kld"][-1] if len(self.metrics["train_kld"]) > 0 else None
                if cur_kld is not None and kld_min <= cur_kld <= kld_max:
                    if self.ckpt_autosave_kld_range_num_files < max_files and \
                            (self.ckpt_autosave_kld_range_last_kld is None or
                             cmp_func(cur_kld - self.ckpt_autosave_kld_range_last_kld, min_change)):
                        if do_valid:
                            self.validate()
                        if self.do_testing:
                            self.test()
                        self.uninterruptible_save(False)
                        self.ckpt_autosave_kld_range_last_kld = cur_kld
                        self.ckpt_autosave_kld_range_num_files += 1

        if self.train_states["train_steps"] % self.opts.validation_rate == 0:
            if "valid_best" in self.opts.ckpt_autosave_approaches:
                metric = self.opts.ckpt_autosave_approaches["valid_best"]
                if isinstance(metric, str):
                    metric = [metric]
                for m in metric:
                    if not hasattr(self, "ckpt_last_best_valid_{}".format(m)):
                        setattr(self, "ckpt_last_best_valid_{}".format(m), None)
                    cur = self.metrics["valid_{}".format(m.lstrip("-"))]["valid"][-1]
                    last = getattr(self, "ckpt_last_best_valid_{}".format(m))
                    if last is None or (m.startswith("-") and cur > last) or (not m.startswith("-") and cur < last):
                        if self.do_testing:
                            self.test()
                        self.uninterruptible_save(only_metrics=False, states_filename="valid_best_{}.pt".format(m))
                        setattr(self, "ckpt_last_best_valid_{}".format(m), cur)

    def init_states(self):
        self.train_states = {
            "model_state": None,
            "optim_state": None,
            "lr_scheduler_state": None,
            "KLD_state": None,
            "opts": None,
            "metrics": None,
            "cur_loss": {"step": 0.0, "mean": 0.0},
            "cur_fwd_ce": {"step": 0.0, "mean": 0.0},
            "cur_bwd_ce": {"step": 0.0, "mean": 0.0},
            "cur_bow_loss": {"step": 0.0, "mean": 0.0},
            "cur_kld": {"step": 0.0, "mean": 0.0},
            "cur_elbo": {"step": 0.0, "mean": 0.0},
            "cur_fwd_ppl": {"step": 0.0, "mean": 0.0},
            "cur_bwd_ppl": {"step": 0.0, "mean": 0.0},
            "train_steps": 0}

    def init_metrics(self):
        self.metrics = {
            "kl_coefficient": [],
            "trnes_loss": [],
            "trnes_fwd_ce": [],
            "trnes_bwd_ce": [],
            "trnes_bow_loss": [],
            "trnes_kld": [],
            "trnes_elbo": [],
            "trnes_fwd_ppl": [],
            "trnes_bwd_ppl": [],
            "train_loss": [],
            "train_fwd_ce": [],
            "train_bwd_ce": [],
            "train_bow_loss": [],
            "train_kld": [],
            "train_elbo": [],
            "train_fwd_ppl": [],
            "train_bwd_ppl": [],
            "valid_loss": {"valid": [], "train": []},
            "valid_fwd_ce": {"valid": [], "train": []},
            "valid_bwd_ce": {"valid": [], "train": []},
            "valid_bow_loss": {"valid": [], "train": []},
            "valid_kld": {"valid": [], "train": []},
            "valid_elbo": {"valid": [], "train": []},
            "valid_fwd_ppl": {"valid": [], "train": []},
            "valid_bwd_ppl": {"valid": [], "train": []},
            "test_metrics": defaultdict(dict),
            "train_steps": [],
            "valid_steps": [],
            "train_time": 0,
            "valid_time": 0}
