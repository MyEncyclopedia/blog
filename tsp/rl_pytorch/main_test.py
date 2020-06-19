import unittest
from copy import deepcopy

import torch
from onmt.translate import BeamSearch

from rl_pytorch.beam_search import Beam



class GlobalScorerStub(object):
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.
        self.cov_penalty = lambda cov, beta: torch.zeros(
            (1, cov.shape[-2]), device=cov.device, dtype=torch.float)
        self.has_cov_pen = False
        self.has_len_pen = False

    def update_global_state(self, beam):
        pass

    def score(self, beam, scores):
        return scores

class TestBeamSearchAgainstReferenceCase(unittest.TestCase):
    # this is just test_beam.TestBeamAgainstReferenceCase repeated
    # in each batch.
    BEAM_SZ = 5
    EOS_IDX = 2  # don't change this - all the scores would need updated
    N_WORDS = 8  # also don't change for same reason
    N_BEST = 3
    DEAD_SCORE = -1e20
    BATCH_SZ = 3
    INP_SEQ_LEN = 53

    def random_attn(self):
        return torch.randn(1, self.BATCH_SZ * self.BEAM_SZ, self.INP_SEQ_LEN)

    def init_step(self, beam, expected_len_pen):
        # init_preds: [4, 3, 5, 6, 7] - no EOS's
        init_scores = torch.log_softmax(torch.tensor(
            [[0, 0, 0, 4, 5, 3, 2, 1]], dtype=torch.float), dim=1)
        init_scores = deepcopy(init_scores.repeat(
            self.BATCH_SZ * self.BEAM_SZ, 1))
        new_scores = init_scores + beam.topk_log_probs.view(-1).unsqueeze(1)
        expected_beam_scores, expected_preds_0 = new_scores \
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS) \
            .topk(self.BEAM_SZ, dim=-1)
        beam.advance(deepcopy(init_scores), self.random_attn())
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_ids.equal(expected_preds_0))
        self.assertFalse(beam.is_finished.any())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def first_step(self, beam, expected_beam_scores, expected_len_pen):
        # no EOS's yet
        assert beam.is_finished.sum() == 0
        scores_1 = torch.log_softmax(torch.tensor(
            [[0, 0, 0, .3, 0, .51, .2, 0],
             [0, 0, 1.5, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, .49, .48, 0, 0],
             [0, 0, 0, .2, .2, .2, .2, .2],
             [0, 0, 0, .2, .2, .2, .2, .2]]
        ), dim=1)
        scores_1 = scores_1.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_1), self.random_attn())

        new_scores = scores_1 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores\
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS)\
            .topk(self.BEAM_SZ, -1)
        expected_bptr_1 = unreduced_preds / self.N_WORDS
        # [5, 3, 2, 6, 0], so beam 2 predicts EOS!
        expected_preds_1 = unreduced_preds - expected_bptr_1 * self.N_WORDS
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_scores.allclose(
            expected_beam_scores / expected_len_pen))
        self.assertTrue(beam.topk_ids.equal(expected_preds_1))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_1))
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        self.assertTrue(beam.is_finished[:, 2].all())  # beam 2 finished
        beam.update_finished()
        self.assertFalse(beam.top_beam_finished.any())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def second_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 2 finished on last step
        scores_2 = torch.log_softmax(torch.tensor(
            [[0, 0, 0, .3, 0, .51, .2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 5000, .48, 0, 0],  # beam 2 shouldn't continue
             [0, 0, 50, .2, .2, .2, .2, .2],  # beam 3 -> beam 0 should die
             [0, 0, 0, .2, .2, .2, .2, .2]]
        ), dim=1)
        scores_2 = scores_2.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_2), self.random_attn())

        # ended beam 2 shouldn't continue
        expected_beam_scores[:, 2::self.BEAM_SZ] = self.DEAD_SCORE
        new_scores = scores_2 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores\
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS)\
            .topk(self.BEAM_SZ, -1)
        expected_bptr_2 = unreduced_preds / self.N_WORDS
        # [2, 5, 3, 6, 0] repeat self.BATCH_SZ, so beam 0 predicts EOS!
        expected_preds_2 = unreduced_preds - expected_bptr_2 * self.N_WORDS
        # [-2.4879, -3.8910, -4.1010, -4.2010, -4.4010] repeat self.BATCH_SZ
        self.assertTrue(beam.topk_log_probs.allclose(expected_beam_scores))
        self.assertTrue(beam.topk_scores.allclose(
            expected_beam_scores / expected_len_pen))
        self.assertTrue(beam.topk_ids.equal(expected_preds_2))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_2))
        # another beam is finished in all batches
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        # new beam 0 finished
        self.assertTrue(beam.is_finished[:, 0].all())
        # new beam 0 is old beam 3
        self.assertTrue(expected_bptr_2[:, 0].eq(3).all())
        beam.update_finished()
        self.assertTrue(beam.top_beam_finished.all())
        self.assertFalse(beam.done)
        return expected_beam_scores

    def third_step(self, beam, expected_beam_scores, expected_len_pen):
        # assumes beam 0 finished on last step
        scores_3 = torch.log_softmax(torch.tensor(
            [[0, 0, 5000, 0, 5000, .51, .2, 0],  # beam 0 shouldn't cont
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 5000, 0, 0],
             [0, 0, 0, .2, .2, .2, .2, .2],
             [0, 0, 50, 0, .2, .2, .2, .2]]  # beam 4 -> beam 1 should die
        ), dim=1)
        scores_3 = scores_3.repeat(self.BATCH_SZ, 1)

        beam.advance(deepcopy(scores_3), self.random_attn())

        expected_beam_scores[:, 0::self.BEAM_SZ] = self.DEAD_SCORE
        new_scores = scores_3 + expected_beam_scores.view(-1).unsqueeze(1)
        expected_beam_scores, unreduced_preds = new_scores\
            .view(self.BATCH_SZ, self.BEAM_SZ * self.N_WORDS)\
            .topk(self.BEAM_SZ, -1)
        expected_bptr_3 = unreduced_preds / self.N_WORDS
        # [5, 2, 6, 1, 0] repeat self.BATCH_SZ, so beam 1 predicts EOS!
        expected_preds_3 = unreduced_preds - expected_bptr_3 * self.N_WORDS
        self.assertTrue(beam.topk_log_probs.allclose(
            expected_beam_scores))
        self.assertTrue(beam.topk_scores.allclose(
            expected_beam_scores / expected_len_pen))
        self.assertTrue(beam.topk_ids.equal(expected_preds_3))
        self.assertTrue(beam.current_backptr.equal(expected_bptr_3))
        self.assertEqual(beam.is_finished.sum(), self.BATCH_SZ)
        # new beam 1 finished
        self.assertTrue(beam.is_finished[:, 1].all())
        # new beam 1 is old beam 4
        self.assertTrue(expected_bptr_3[:, 1].eq(4).all())
        beam.update_finished()
        self.assertTrue(beam.top_beam_finished.all())
        self.assertTrue(beam.done)
        return expected_beam_scores

    def test_beam_advance_against_known_reference(self):
        beam = BeamSearch(
            self.BEAM_SZ, self.BATCH_SZ, 0, 1, 2, self.N_BEST,
            GlobalScorerStub(),
            0, 30, False, 0, set(),
            False, 0.)
        device_init = torch.zeros(1, 1)
        beam.initialize(device_init, torch.randint(0, 30, (self.BATCH_SZ,)))
        expected_beam_scores = self.init_step(beam, 1)
        expected_beam_scores = self.first_step(beam, expected_beam_scores, 1)
        expected_beam_scores = self.second_step(beam, expected_beam_scores, 1)
        self.third_step(beam, expected_beam_scores, 1)


if __name__ == "__main__":
    # initial = [0.35, 0.25, 0.4]
    #
    # transition = [
    #     [0.1, 0.1, 0.8],
    #     [0.3, 0.6, 0.1],
    #     [0.4, 0.2, 0.4],
    #     [0.3, 0.4, 0.4],
    # ]
    #
    # b = Beam(3, 4)
    # # print(b.get_current_state())
    # # print(b.get_current_origin())
    # # b.advance(torch.FloatTensor(transition[0]))
    # print(b.advance(torch.FloatTensor(transition)))
    # print(b.advance(torch.FloatTensor(transition)))
    # print(b.advance(torch.FloatTensor(transition)))
    # print(b.get_current_state())
    # print(b.get_current_origin())
    # print(b.get_best())
    # print(b.get_hyp(0))
    c = TestBeamSearchAgainstReferenceCase()
    c.test_beam_advance_against_known_reference()
