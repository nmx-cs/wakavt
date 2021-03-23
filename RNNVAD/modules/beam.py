# -*- coding: utf-8 -*-


from torch.nn import functional


# Code adapted from https://github.com/OpenNMT/OpenNMT-py/blob/eaade1dc05963c1cc638c0cf6f3f714d6a555c0e/onmt/translate/beam.py
class Beam(object):
    def __init__(self, beam_width, length_norm, EOS_token, n_best, prefix_len=0):
        self.beam_width = beam_width
        self.length_norm = length_norm
        self.EOS_token = EOS_token
        self.n_best = n_best
        self.prefix_len = prefix_len
        assert self.n_best <= self.beam_width

        self.beams = []
        self.prev_idx = []
        self.beam_scores = None
        self.finished_beams = []
        self.eos_top = False

    def _update_stats(self, scores, word_idx, beam_idx=None):
        self.beams.append(word_idx.tolist())
        if beam_idx is not None:
            self.prev_idx.append(beam_idx.tolist())
        self.beam_scores = scores

        scores = scores.tolist()
        seq_len = len(self.beams)
        for i in range(self.beam_width):
            if self.beams[-1][i] == self.EOS_token:
                normed_score = self.length_penalty(scores[i], seq_len)
                self.finished_beams.append((i, seq_len - 1, normed_score))

        if self.beams[-1][0] == self.EOS_token:
            self.eos_top = True

        assert len(self.beams) == len(self.prev_idx) + 1, "beam state error"

    # logits : [vocab_size] or [beam_width, vocab_size]
    def init_beams(self, logits):
        assert len(self.beams) == len(self.prev_idx) == 0, "init_beams must be called only once"
        assert logits.dim() in (1, 2)
        vocab_size = logits.size(-1)
        if logits.dim() == 2:
            assert logits.size(0) == self.beam_width
        log_probs = functional.log_softmax(logits, dim=-1)
        top_probs, word_idx = log_probs.view(-1).topk(self.beam_width, dim=0, largest=True, sorted=True)
        beam_idx = None

        if logits.dim() == 2:
            word_idx = word_idx % vocab_size
            beam_idx = word_idx // vocab_size
        self._update_stats(top_probs, word_idx)

        return word_idx, beam_idx

    # Please refer to https://arxiv.org/pdf/1609.08144.pdf  Section 7
    def length_penalty(self, score, seq_len):
        modifier = ((5 + seq_len + self.prefix_len) ** self.length_norm) / ((5 + 1) ** self.length_norm)
        return score / modifier

    # logits : [beam_width, vocab_size]
    def update_beams(self, logits):
        assert len(self.beams) > 0, "init_beams must be called before calling update_beams"
        assert logits.dim() == 2
        vocab_size = logits.size(1)

        log_probs = functional.log_softmax(logits, dim=1)
        scores = log_probs + self.beam_scores.unsqueeze(1).expand_as(log_probs)

        # make sure EOS won't be expanded
        for i in range(self.beam_width):
            if self.beams[-1][i] == self.EOS_token:
                scores[i] = -1e20

        best_scores, best_idx = scores.view(-1).topk(self.beam_width, dim=0, largest=True, sorted=True)
        word_idx = best_idx % vocab_size
        beam_idx = best_idx // vocab_size
        self._update_stats(best_scores, word_idx, beam_idx)

        return word_idx, beam_idx

    @property
    def done(self):
        return self.eos_top and len(self.finished_beams) >= self.n_best

    # Walk back to construct the full hypothesis. 
    def get_hyp(self, beam_idx, timestep, leftward):
        hyp = []
        k = beam_idx
        for i in range(timestep, 0, -1):
            hyp.append(self.beams[i][k])
            k = self.prev_idx[i - 1][k]
        hyp.append(self.beams[0][k])
        return hyp[::(1 if leftward else -1)], k

    def sort_finished(self, add_unfinished=False):
        if add_unfinished:
            seq_len = len(self.beams)
            for i in range(self.beam_width):
                if self.beams[-1][i] != self.EOS_token:
                    normed_score = self.length_penalty(self.beam_scores[i].tolist(), seq_len)
                    self.finished_beams.append((i, seq_len - 1, normed_score))
        self.finished_beams.sort(key=lambda s: s[-1], reverse=True)

    def get_best_results(self, leftward=False):
        add_unfinished = True if len(self.finished_beams) < self.n_best else False
        self.sort_finished(add_unfinished)

        results = []
        for i in range(self.n_best):
            beam_idx, timestep, _ = self.finished_beams[i]
            results.append(self.get_hyp(beam_idx, timestep, leftward))
        results, init_beam_idx = zip(*results)
        return list(results), list(init_beam_idx)


def step_batch_beams(batch_beams, logits_step, output_step, back_pointers=None, func="update_beams"):
    assert func in ("init_beams", "update_beams")
    batch_size = len(batch_beams)
    beam_width = output_step.size(0) // batch_size
    for i in range(batch_size):
        word_idx, beam_idx = getattr(batch_beams[i], func)(logits_step[i])
        output_step[i * beam_width: (i + 1) * beam_width] = word_idx
        if beam_idx is not None and back_pointers is not None:
            back_pointers[i * beam_width: (i + 1) * beam_width] = beam_idx + i * beam_width
