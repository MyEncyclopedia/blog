from copy import deepcopy
from math import exp
import torch
from onmt.translate import BeamSearch, GNMTGlobalScorer

def run_example():
    BEAM_SIZE = 2
    N_BEST = 1
    BATCH_SZ = 1
    SEQ_LEN = 3

    transition = [
        [0.3, 0.6, 0.1],
        [0.4, 0.2, 0.4],
        [0.3, 0.4, 0.4]]

    beam = BeamSearch(BEAM_SIZE, BATCH_SZ, 0, 1, 2, N_BEST, GNMTGlobalScorer(0.7, 0., "avg", "none"), 0, 30, False, 0, set(), False, 0.)
    device_init = torch.zeros(1, 1)
    beam.initialize(device_init, torch.randint(0, 30, (BATCH_SZ,)))

    init_scores = torch.log(torch.tensor([[0.35, 0.25, 0.4]], dtype=torch.float))
    init_scores = deepcopy(init_scores.repeat(BATCH_SZ * BEAM_SIZE, 1))
    beam.advance(init_scores, None)

    for step in range(SEQ_LEN - 1):
        idx_list = beam.topk_ids.squeeze().tolist()
        beam_transition = []
        for idx in idx_list:
            beam_transition.append(transition[idx])
        beam_transition_tensor = torch.log(torch.tensor(beam_transition))

        beam.advance(beam_transition_tensor, None)
        beam.update_finished()

        print(f'step {step} beam results')
        for k in range(BEAM_SIZE):
            best_path = beam.alive_seq[k].squeeze().tolist()[1:]
            prob = exp(beam.topk_log_probs[0][k])
            print(f'prob {prob:.3f} with path {best_path}')


if __name__ == "__main__":
    run_example()

