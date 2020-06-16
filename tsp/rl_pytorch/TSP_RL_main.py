import math
from typing import List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
import torch.autograd as autograd
import torch.nn.functional as F
from IPython.core.display import clear_output
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from tqdm import tqdm

USE_CUDA = False

# todo:
# 1. move use_cuda out
# 2. configurable embedding
# 3. unify data set
# 4. beam search decoder
# 5. critic net


class TSPUnlabeledDataset(Dataset):

    def __init__(self, num_nodes, num_samples, random_seed=9):
        super(TSPUnlabeledDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for _ in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau'):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query: Tensor, ref: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   [batch_size x seq_len x hidden_size]
        Returns:
            ref:    [batch_size x hidden_size, seq_len]
            logits: [batch_size x seq_len]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, torch.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * torch.tanh(logits)
        else:
            logits = logits
        return ref, logits


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size

        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, batch_input: Tensor) -> Tensor:
        """

        Args:
            batch_input: [batch_size x 2 x seq_len]
        Returns:
            embedded: [batch_size, input_size, embedding_size]

        """
        batch_size = batch_input.size(0)
        seq_len = batch_input.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded = []
        batch_input = batch_input.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(batch_input[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded


class PointerNet(nn.Module):
    def __init__(self, embedding_size, hidden_size, seq_len, num_glimpse, tanh_exploration, use_tanh, attention):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_glimpse = num_glimpse
        self.seq_len = seq_len

        self.embedding = GraphEmbedding(2, embedding_size)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits: Tensor, mask: Tensor, idxs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logits: [batch_size x seq_len]
            mask:   [batch_size x seq_len]
            idxs:   None or tensor [batch_size]
        Returns:
            logits:      []
            mask_clone:  []
        """
        batch_size = logits.size(0)
        mask_clone = mask.clone()

        if idxs is not None:
            mask_clone[[i for i in range(batch_size)], idxs.data] = 1
            logits[mask_clone.bool()] = -np.inf
        return logits, mask_clone

    def forward(self, batch_input: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            batch_input: [batch_size x 2 x seq_len]
        Returns:
            prob_list:        [batch_size x seq_len][seq_len]
            action_idx_list:  [batch_size][seq_len]
        """
        batch_size = batch_input.size(0)
        seq_len = batch_input.size(2)

        embedded = self.embedding(batch_input)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prob_list = []
        action_idx_list = []
        mask = torch.zeros(batch_size, seq_len).byte()

        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)
            for i in range(self.num_glimpse):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits)

            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in action_idx_list:
                if old_idxs.eq(idxs).data.any():
                    print(f'{seq_len}')
                    print(' RESAMPLE!')
                    idxs = probs.multinomial(1).squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]

            prob_list.append(probs)
            action_idx_list.append(idxs)

        return prob_list, action_idx_list


class CombinatorialRL(nn.Module):
    def __init__(self, embedding_size, hidden_size, seq_len, num_glimpse, tanh_exploration, use_tanh, attention):
        super(CombinatorialRL, self).__init__()

        self.actor = PointerNet(embedding_size, hidden_size, seq_len, num_glimpse, tanh_exploration, use_tanh, attention)

    def forward(self, batch_input: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]:
        """
        Args:
            batch_input: [batch_size, 2, seq_len]
        Returns:
            R: Tensor of shape 32
            action_prob_list: [batch_size][seq_len]
            action_list:      [batch_size x 2][seq_len]
            action_idx_list:  [batch_size][seq_len]
        """
        batch_size = batch_input.size(0)
        seq_len = batch_input.size(2)

        prob_list, action_idx_list = self.actor(batch_input)

        action_list = []
        batch_input = batch_input.transpose(1, 2)
        for action_id in action_idx_list:
            action_list.append(batch_input[[x for x in range(batch_size)], action_id.data, :])
        action_prob_list = []
        for prob, action_id in zip(prob_list, action_idx_list):
            action_prob_list.append(prob[[x for x in range(batch_size)], action_id.data])

        R = self.reward(action_list)

        return R, action_prob_list, action_list, action_idx_list


    def reward(self, sample_solution: List[Tensor]) -> Tensor:
        """
        Computes total distance of tour
        Args:
            sample_solution: list of size N, each tensor of shape [batch_size x 2]

        Returns:
            tour_len: [32]

        """
        batch_size = sample_solution[0].size(0)
        n = len(sample_solution)
        tour_len = Variable(torch.zeros([batch_size]))

        for i in range(n - 1):
            tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)
        tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)
        return tour_len



if __name__ == "__main__":
    train_size = 100000
    validate_size = 1000
    train_dataset = TSPUnlabeledDataset(10, train_size)
    validate_dataset = TSPUnlabeledDataset(10, validate_size)

    embedding_size = 128
    hidden_size = 128
    num_glimpse = 1
    tanh_exploration = 10
    use_tanh = True

    beta = 0.9
    max_grad_norm = 2.

    RL_model = CombinatorialRL(embedding_size, hidden_size, 10, num_glimpse, tanh_exploration, use_tanh, attention="Dot")

    batch_size = 32
    threshold = 3.99
    max_grad_norm = 2.0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    actor_optim = optim.Adam(RL_model.actor.parameters(), lr=1e-4)

    epochs = 0

    critic_exp_mvg_avg = torch.zeros(1)

    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_id, sample_batch in enumerate(train_loader):
            train_tour = []
            # print(f'{epoch}: {batch_id}')
            RL_model.train()

            batch_input = Variable(sample_batch)

            R, prob_list, action_list, actions_idx_list = RL_model(batch_input)

            if batch_id == 0:
                critic_exp_mvg_avg = R.mean()
            else:
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

            advantage = R - critic_exp_mvg_avg

            log_probs = 0
            for prob in prob_list:
                log_prob = torch.log(prob)
                log_probs += log_prob
            log_probs[log_probs < -1000] = 0.

            reinforce = advantage * log_probs
            actor_loss = reinforce.mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm(RL_model.actor.parameters(), float(max_grad_norm), norm_type=2)

            actor_optim.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            train_tour.append(R.mean().item())

            if batch_id % 100 == 0:
                validate_tour = []
                RL_model.eval()
                for validate_batch in validate_loader:
                    batch_input_validate = Variable(validate_batch)

                    R, prob_list, action_list, actions_idx_list = RL_model(batch_input_validate)
                    validate_tour.append(R.mean().item())
                print(f'{epoch} : {batch_id}')
                print('validate tour {}'.format(sum(validate_tour) / len(validate_tour)))
                print('train tour {}'.format(sum(train_tour) / len(train_tour)))

        # if threshold and train_tour[-1] < threshold:
        #     print("EARLY STOP!")
        #     break

        epochs += 1

