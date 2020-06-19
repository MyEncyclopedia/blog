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

from rl_pytorch.TSP_dataset import TSPDataset, TSPUnlabeledDataset
from rl_pytorch.beam_search import Beam

USE_CUDA = False

# todo:
# 1. bidirectional True does not work
# 2. beam search decoder
# 3. critic net
# 4. plot
# 5. mask
# 6. use_cuda gpu
# https://medium.com/the-artificial-impostor/implementing-beam-search-part-1-4f53482daabe

def rnn_init(rnn_type: str, **kwargs) -> nn.RNNBase:
    if rnn_type in ["LSTM", "GRU", "RNN"]:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn

class Attention(nn.Module):
    use_tanh: bool
    C: int
    name: str

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
            query: [batch_size * hidden_size]
            ref:   [batch_size * seq_len * hidden_size]
        Returns:
            ref:    [batch_size * hidden_size * seq_len]
            logits: [batch_size * seq_len]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size * hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size * seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size * hidden_size * seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size * 1 * hidden_size]
            logits = torch.bmm(V, torch.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size * seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * torch.tanh(logits)
        else:
            logits = logits
        return ref, logits


class AttentionB(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10):
        super(AttentionB, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        v = torch.FloatTensor(dim)
        self.v = nn.Parameter(v)
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits

class Encoder(nn.Module):
    """Maps a graph represented as an input sequence
    to a hidden vector"""

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.enc_init_state = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=False)

        # enc_init_hx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))

        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=False)

        # enc_init_cx = nn.Parameter(enc_init_cx)
        # enc_init_cx.data.uniform_(-(1. / math.sqrt(hidden_dim)),
        #        1. / math.sqrt(hidden_dim))
        return (enc_init_hx, enc_init_cx)



class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 max_length,
                 tanh_exploration,
                 terminating_symbol,
                 use_tanh,
                 decode_type,
                 n_glimpses=1,
                 beam_size=0,
                 ):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.max_length = max_length
        self.terminating_symbol = terminating_symbol
        self.decode_type = decode_type
        self.beam_size = beam_size

        self.input_weights = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_weights = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.pointer = AttentionB(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = AttentionB(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax()

    def apply_mask_to_logits(self, step, logits, mask, prev_idxs):
        if mask is None:
            mask = torch.zeros(logits.size()).byte()

        maskk = mask.clone()

        # to prevent them from being reselected.
        # Or, allow re-selection and penalize in the objective function
        if prev_idxs is not None:
            # set most recently selected idx values to 1
            maskk[[x for x in range(logits.size(0))],
                  prev_idxs.data] = 1
            logits[maskk] = -np.inf
        return logits, maskk

    def forward(self, decoder_input, embedded_inputs, hidden, context):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        def recurrence(x, hidden, logit_mask, prev_idxs, step):

            hx, cx = hidden  # batch_size x hidden_dim

            gates = self.input_weights(x) + self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # batch_size x hidden_dim

            g_l = hy
            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(g_l, context)
                logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
                # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
                # [batch_size x h_dim x 1]
                g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(g_l, context)

            logits, logit_mask = self.apply_mask_to_logits(step, logits, logit_mask, prev_idxs)
            probs = self.sm(logits)
            return hy, cy, probs, logit_mask

        batch_size = context.size(1)
        outputs = []   # [seq_len][batch_size * seq_len]
        selections = []  # [seq_len][batch_size]
        steps = range(self.max_length)  # or until terminating symbol ?
        inps = []
        idxs = None
        mask = None

        if self.decode_type == "stochastic":
            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
                hidden = (hx, cx)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs = self.decode_stochastic(probs, embedded_inputs, selections)
                inps.append(decoder_input)
                # use outs to point to next object
                outputs.append(probs)
                selections.append(idxs)
            return (outputs, selections), hidden

        elif self.decode_type == "beam_search":

            # Expand input tensors for beam search
            decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))
            context = Variable(context.data.repeat(1, self.beam_size, 1))
            hidden = (Variable(hidden[0].data.repeat(self.beam_size, 1)),
                      Variable(hidden[1].data.repeat(self.beam_size, 1)))

            beam = [
                Beam(self.beam_size, self.max_length)
                for k in range(batch_size)
            ]

            for i in steps:
                hx, cx, probs, mask = recurrence(decoder_input, hidden, mask, idxs, i)
                hidden = (hx, cx)

                probs = probs.view(self.beam_size, batch_size, -1).transpose(0, 1).contiguous()

                n_best = 1
                # select the next inputs for the decoder [batch_size x hidden_dim]
                decoder_input, idxs, active = self.decode_beam(probs, embedded_inputs, beam, batch_size, n_best, i)

                inps.append(decoder_input)
                # use probs to point to next object
                if self.beam_size > 1:
                    outputs.append(probs[:, 0, :])
                else:
                    outputs.append(probs.squeeze(0))
                # Check for indexing
                selections.append(idxs)
                # Should be done decoding
                if len(active) == 0:
                    break
                decoder_input = Variable(decoder_input.data.repeat(self.beam_size, 1))

            return (outputs, selections), hidden

    def decode_stochastic(self, probs, embedded_inputs, selections):
        """
        Return the next input for the decoder by selecting the
        input corresponding to the max output

        Args:
            probs: [batch_size x sourceL]
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            selections: list of all of the previously selected indices during decoding
       Returns:
            Tensor of size [batch_size x sourceL] containing the embeddings
            from the inputs corresponding to the [batch_size] indices
            selected for this iteration of the decoding, as well as the
            corresponding indicies
        """
        batch_size = probs.size(0)
        # idxs is [batch_size]
        idxs = probs.multinomial(1).squeeze(1)

        # due to race conditions, might need to resample here
        for old_idxs in selections:
            # compare new idxs
            # elementwise with the previous idxs. If any matches,
            # then need to resample
            if old_idxs.eq(idxs).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial(1).squeeze(1)
                break

        sels = embedded_inputs[idxs.data, [i for i in range(batch_size)], :]  # [batch_size * embedding_size]
        return sels, idxs

    def decode_beam(self, probs, embedded_inputs, beam, batch_size, n_best, step):
        active = []
        for b in range(batch_size):
            if beam[b].done:
                continue

            if not beam[b].advance(probs.data[b]):
                active += [b]

        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]

        all_idxs = Variable(torch.LongTensor([[x for x in hyp] for hyp in all_hyp]).squeeze())  # [batch_size]
        print(f'all_idx.dim {all_idxs.dim()}')
        if all_idxs.dim() == 2:
            if all_idxs.size(1) > n_best:
                idxs = all_idxs[:, -1]
            else:
                idxs = all_idxs
        elif all_idxs.dim() == 3:
            idxs = all_idxs[:, -1, :]
        else:
            if all_idxs.size(0) > 1:
                idxs = all_idxs[-1]
            else:
                idxs = all_idxs

        if idxs.dim() > 1:
            x = embedded_inputs[idxs.transpose(0, 1).contiguous().data,
                [x for x in range(batch_size)], :]
            return x.view(idxs.size(0) * n_best, embedded_inputs.size(2)), idxs, active
        else:
            x = embedded_inputs[idxs.data, [x for x in range(batch_size)], :]  # [batch_size * embedding_size]
            return x, idxs, active


class PointerNetwork(nn.Module):
    """The pointer network, which is the core seq2seq
    model"""

    def __init__(self,
                 decode_type,
                 embedding_dim,
                 hidden_dim,
                 max_decoding_len,
                 terminating_symbol,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 beam_size,
                 ):
        super(PointerNetwork, self).__init__()

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
            )

        self.decoder = Decoder(
            embedding_dim,
            hidden_dim,
            max_length=max_decoding_len,
            tanh_exploration=tanh_exploration,
            use_tanh=use_tanh,
            terminating_symbol=terminating_symbol,
            decode_type=decode_type,
            n_glimpses=n_glimpses,
            beam_size=beam_size,
        )

        # Trainable initial hidden states
        dec_in_0 = torch.FloatTensor(embedding_dim)

        self.decoder_in_0 = nn.Parameter(dec_in_0)
        self.decoder_in_0.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                                        1. / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """ Propagate inputs through the network
        Args:
            inputs: [sourceL x batch_size x embedding_dim]
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_h, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        dec_init_state = (enc_h_t[-1], enc_c_t[-1])

        # repeat decoder_in_0 across batch
        decoder_input = self.decoder_in_0.unsqueeze(0).repeat(inputs.size(1), 1)

        (pointer_probs, input_idxs), dec_hidden_t = self.decoder(decoder_input,
                                                                 inputs,
                                                                 dec_init_state,
                                                                 enc_h)

        return pointer_probs, input_idxs


class CriticNetwork(nn.Module):
    """Useful as a baseline in REINFORCE updates"""

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_process_block_iters,
                 tanh_exploration,
                 use_tanh,
                 ):
        super(CriticNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_process_block_iters = n_process_block_iters

        self.encoder = Encoder(
            embedding_dim,
            hidden_dim,
        )

        self.process_block = AttentionB(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.sm = nn.Softmax()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [embedding_dim x batch_size x sourceL] of embedded inputs
        """

        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inputs.size(1), 1).unsqueeze(0)

        # encoder forward pass
        enc_outputs, (enc_h_t, enc_c_t) = self.encoder(inputs, (encoder_hx, encoder_cx))

        # grab the hidden state and process it via the process block
        process_block_state = enc_h_t[-1]
        for i in range(self.n_process_block_iters):
            ref, logits = self.process_block(process_block_state, enc_outputs)
            process_block_state = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        # produce the final scalar output
        out = self.decoder(process_block_state)
        return out


class NeuralCombinatorialRL(nn.Module):
    """
    This module contains the PointerNetwork (actor) and
    CriticNetwork (critic). It requires
    an application-specific reward function
    """

    def __init__(self,
                 decode_type,
                 embedding_dim,
                 hidden_dim,
                 max_decoding_len,
                 terminating_symbol,
                 n_glimpses,
                 n_process_block_iters,
                 tanh_exploration,
                 use_tanh,
                 beam_size,
                 ):
        super(NeuralCombinatorialRL, self).__init__()

        self.actor = PointerNetwork(
            decode_type,
            embedding_dim,
            hidden_dim,
            max_decoding_len,
            terminating_symbol,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            beam_size,
            )

        # self.critic = CriticNetwork(
        #        embedding_dim,
        #        hidden_dim,
        #        n_process_block_iters,
        #        tanh_exploration,
        #        False)


        embedding_ = torch.FloatTensor(2, embedding_dim)
        self.embedding = nn.Parameter(embedding_)
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_dim)),
                                     1. / math.sqrt(embedding_dim))

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_dim, sourceL]
        """
        print('rl forward')
        batch_size = inputs.size(0)
        input_dim = inputs.size(1)
        sourceL = inputs.size(2)

        # repeat embeddings across batch_size
        # result is [batch_size x input_dim x embedding_dim]
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded_inputs = []
        # result is [batch_size, 1, input_dim, sourceL]
        ips = inputs.unsqueeze(1)

        for i in range(sourceL):
            # [batch_size x 1 x input_dim] * [batch_size x input_dim x embedding_dim]
            # result is [batch_size, embedding_dim]
            embedded_inputs.append(torch.bmm(
                ips[:, :, :, i].float(),
                embedding).squeeze(1))

        # Result is [sourceL x batch_size x embedding_dim]
        embedded_inputs = torch.cat(embedded_inputs).view(
            sourceL,
            batch_size,
            embedding.size(2))

        # query the actor net for the input indices
        # making up the output, and the pointer attn
        probs_, action_idxs = self.actor(embedded_inputs)

        # Select the actions (inputs pointed to
        # by the pointer net) and the corresponding
        # logits
        # should be size [batch_size x
        actions = []
        # inputs is [batch_size, input_dim, sourceL]
        inputs_ = inputs.transpose(1, 2)
        # inputs_ is [batch_size, sourceL, input_dim]
        for action_id in action_idxs:
            actions.append(inputs_[[x for x in range(batch_size)], action_id.data, :])

        if self.training:
            # probs_ is a list of len sourceL of [batch_size x sourceL]
            probs = []
            for prob, action_id in zip(probs_, action_idxs):
                print(f'{prob.shape} {action_id.data}')
                probs.append(prob[[x for x in range(batch_size)], action_id.data])
        else:
            # return the list of len sourceL of [batch_size x sourceL]
            probs = probs_

        # get the critic value fn estimates for the baseline
        # [batch_size]
        # v = self.critic_net(embedded_inputs)

        # [batch_size]
        R = self.reward(actions)

        # return R, v, probs, actions, action_idxs
        return R, probs, actions, action_idxs

    def reward(self, sample_solution: List[Tensor]) -> Tensor:
        """
        Computes total distance of tour
        Args:
            sample_solution: list of size N, each tensor of shape [batch_size*2]

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

class GraphEmbedding(nn.Module):
    embedding: nn.Parameter

    def __init__(self, input_size, embedding_size):
        super(GraphEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, batch_input: Tensor) -> Tensor:
        """

        Args:
            batch_input: [batch_size * 2 * seq_len]
        Returns:
            embedded: [batch_size * input_size * embedding_size]

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
    use_embedding: bool
    embedding: GraphEmbedding
    num_glimpse: int
    encoder: nn.RNNBase
    decoder: nn.RNNBase
    ptr_net: Attention
    glimpse: Attention
    decoder_start_input: nn.Parameter

    def __init__(self, rnn_type, use_embedding, embedding_size, hidden_size, seq_len, num_glimpse, tanh_exploration, use_tanh, attention):
        super(PointerNet, self).__init__()

        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = GraphEmbedding(2, embedding_size)
        else:
            embedding_size = 2

        self.num_glimpse = num_glimpse
        self.encoder = rnn_init(rnn_type, input_size = embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.decoder = rnn_init(rnn_type, input_size = embedding_size, hidden_size=hidden_size, batch_first=True, bidirectional=False)
        self.ptr_net = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits: Tensor, mask: Tensor, idxs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            logits: [batch_size * seq_len]
            mask:   [batch_size * seq_len]
            idxs:   None or tensor [batch_size]
        Returns:
            logits:      []
            mask_clone:  []
        """
        batch_size = logits.size(0)
        mask_clone = mask.clone()

        if idxs is not None:
            mask_clone[[i for i in range(batch_size)], idxs.data] = 1
            logits[mask_clone] = -np.inf
        return logits, mask_clone

    def forward(self, batch_input: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            batch_input: [batch_size * 2 * seq_len]
        Returns:
            prob_list:        [batch_size * seq_len][seq_len]
            action_idx_list:  [batch_size][seq_len]
        """
        batch_size = batch_input.size(0)
        seq_len = batch_input.size(2)

        if self.use_embedding:
            embedded = self.embedding(batch_input)  # [batch_size * seq_len * embedded_size]
        else:
            embedded = batch_input.permute(0, 2, 1)  # [batch_size * seq_len * embedded_size]

        encoder_outputs, hidden = self.encoder(embedded)

        prob_list = []
        action_idx_list = []
        mask = torch.zeros(batch_size, seq_len).byte()
        # mask = torch.zeros(batch_size, seq_len).bool()

        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):
            _, hidden = self.decoder(decoder_input.unsqueeze(1), hidden)

            if isinstance(hidden, tuple):
                query = hidden[0].squeeze(0)
            else:
                query = hidden.squeeze(0)
            for i in range(self.num_glimpse):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)

            _, logits = self.ptr_net(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits, dim=1)

            idxs = probs.multinomial(1).squeeze(1)  # [batch_size]
            for old_idxs in action_idx_list:
                if old_idxs.eq(idxs).data.any():
                    print(f'{seq_len} resample')
                    idxs = probs.multinomial(1).squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]  # [batch_size * embedded_size]

            prob_list.append(probs)
            action_idx_list.append(idxs)

        return prob_list, action_idx_list


class CombinatorialRL(nn.Module):
    actor: PointerNet

    def __init__(self, rnn_type, use_embedding, embedding_size, hidden_size, seq_len, num_glimpse, tanh_exploration, use_tanh, attention):
        super(CombinatorialRL, self).__init__()

        self.actor = PointerNet(rnn_type, use_embedding, embedding_size, hidden_size, seq_len, num_glimpse, tanh_exploration, use_tanh, attention)

    def forward(self, batch_input: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor]]:
        """
        Args:
            batch_input: [batch_size * 2 * seq_len]
        Returns:
            R: Tensor of shape 32
            action_prob_list: [batch_size][seq_len]
            action_list:      [batch_size*2][seq_len]
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
            sample_solution: list of size N, each tensor of shape [batch_size*2]

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
    import argparse

    parser = argparse.ArgumentParser("TSP_RL")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--random_train_size", type=int, default=100000)
    parser.add_argument("--random_validate_size", type=int, default=1000)
    parser.add_argument("--num_glimpse", type=int, default=1)
    parser.add_argument("--use_embedding", type=int, default=1)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5)
    parser.add_argument("--train_filename", type=str, default="../tsp_10_test_sample.txt")
    parser.add_argument("--validate_filename", type=str, default="../tsp_10_test_sample.txt")
    parser.add_argument("--clip_norm", type=float, default=2.)
    parser.add_argument("--threshold", type=float, default=3.99)
    parser.add_argument("--rnn_type", type=str, default='GRU')
    parser.add_argument("--log_dir", type=str, default="./log")

    parser.add_argument('--terminating_symbol', default='<0>', help='')
    parser.add_argument('--n_process_blocks', default=3,
                        help='Number of process block iters to run in the Critic network')
    parser.add_argument('--beam_size', default=1, help='Beam width for beam search')
    parser.add_argument('--dropout', default=0., help='')
    # parser.add_argument('--decode_type', type=str, default='stochastic', help='')
    parser.add_argument('--decode_type', type=str, default='beam_search', help='')

    args = parser.parse_args()

    tanh_exploration = 10
    use_tanh = True
    beta = 0.9

    # RL_model = CombinatorialRL(args.rnn_type, args.use_embedding, args.embedding_size, args.hidden_size, 10, args.num_glimpse, tanh_exploration, use_tanh, attention="Dot")
    RL_model = NeuralCombinatorialRL(args.decode_type, args.embedding_size, args.hidden_size, 10, args.terminating_symbol, args.num_glimpse,
                                     args.n_process_blocks, tanh_exploration, use_tanh, args.beam_size)

    use_random_ds = True
    if use_random_ds:
        train_dataset = TSPUnlabeledDataset(10, args.random_train_size)
        validate_dataset = TSPUnlabeledDataset(10, args.random_validate_size)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        validate_loader = DataLoader(validate_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    else:
        train_ds = TSPDataset(args.train_filename, 10, 10)
        test_ds = TSPDataset(args.validate_file, 10, 10)
        train_loader = DataLoader(train_ds, num_workers=0, batch_size=args.batch_size)
        validate_loader = DataLoader(test_ds, num_workers=0, batch_size=args.batch_size)

    actor_optim = optim.Adam(RL_model.actor.parameters(), lr=1e-4)
    critic_exp_mvg_avg = torch.zeros(1)
    threshold_stop = False

    for epoch in range(args.num_epoch):
        batch_id = 0
        for batch_item in train_loader:
            batch_input = batch_item[0] # [batch_size * 2 * seq_len]
            batch_id += 1
            train_tour = []
            print(f'{epoch}: {batch_id}')
            RL_model.train()

            batch_input = Variable(batch_input)

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
            torch.nn.utils.clip_grad_norm_(RL_model.actor.parameters(), float(args.clip_norm), norm_type=2)

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

                validate_tour_avg_r = sum(validate_tour) / len(validate_tour)
                train_tour_batch_avg_r = sum(train_tour) / len(train_tour)
                print(f'{epoch} : {batch_id}')
                print(f'validate tour {validate_tour_avg_r}')
                print(f'train tour {train_tour_batch_avg_r}')

                if args.threshold and validate_tour_avg_r < args.threshold:
                    threshold_stop = True
                    print("EARLY STOP!")
                    break
            if threshold_stop:
                break;


