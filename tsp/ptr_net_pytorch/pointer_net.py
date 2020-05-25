# coding=utf-8
from typing import Tuple, Union

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F


def rnn_init(rnn_type: str, **kwargs) -> nn.RNNBase:
    if rnn_type in ["LSTM", "GRU", "RNN"]:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn


def sequence_mask(lengths: Tensor, max_len: int = None) -> Tensor:
    """
    Create mask

    Args:
        lengths (LongTensor) : lengths (batch_size)
        max_len (int) : maximum length

    Returns:
      mask (batch_size, max_len)
    """

    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths))


class Attention(nn.Module):
    linear_out: nn.Linear

    def __init__(self, dim: int):
        """
        Attention layer
        Args:
            dim: input dimension size
        """
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)

    def score(self, src: Tensor, target: Tensor) -> Tensor:
        """
        Attention score calculation

        Args:
            src: source values (batch_size, src_len, dim)
            target: target values (batch_size, target_len, dim)

        Returns:

        """

        batch_size, src_len, dim = src.size()
        _, target_len, _ = target.size()

        target_ = target
        src_ = src.transpose(1, 2)
        return torch.bmm(target_, src_)

    def forward(self, src: Tensor, target: Tensor, src_lengths: Tensor = None) -> Tuple[Tensor, Tensor]:
        """

        Args:
            src: source values (batch_size, src_len, dim)
            target: target values (batch_size, target_len, dim)
            src_lengths: source values length

        Returns:

        """
        assert target.dim() == 3

        batch_size, src_len, dim = src.size()
        _, target_len, _ = target.size()

        align_score = self.score(src, target)

        if src_lengths is not None:
            mask = sequence_mask(src_lengths)
            # (batch_size, max_len) -> (batch_size, 1, max_len)
            # so mask can broadcast
            mask = mask.unsqueeze(1)
            align_score.data.masked_fill_(~mask, -float('inf'))

        # normalize weights
        align_score = F.softmax(align_score, -1)

        c = torch.bmm(align_score, src)

        concat_c = torch.cat([c, target], -1)
        attn_h = self.linear_out(concat_c)

        return attn_h, align_score


class RNNEncoder(nn.Module):
    rnn: Union[nn.LSTM, nn.GRU, nn.RNN]

    def __init__(self, rnn_type: str, bidirectional: bool, num_layers: int, input_size: int, hidden_size: int,
                 dropout: float):
        """

        Args:
            rnn_type: rnn cell type, ["LSTM", "GRU", "RNN"]
            bidirectional: whether use bidirectional rnn
            num_layers: number of layers in stacked rnn
            input_size: input dimension size
            hidden_size: rnn hidden dimension size
            dropout: dropout rate
        """

        super(RNNEncoder, self).__init__()
        if bidirectional:
            assert hidden_size % 2 == 0
            hidden_size = hidden_size // 2
        self.rnn = rnn_init(rnn_type, input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional,
                            num_layers=num_layers, dropout=dropout)

    def forward(self, src: Tensor, lengths: Tensor = None, hidden: Tensor = None) -> Tuple[Tensor, Tensor]:
        lengths = lengths.view(-1).tolist()
        packed_src = pack_padded_sequence(src, lengths)
        memory_bank, hidden_final = self.rnn(packed_src, hidden)
        memory_bank = pad_packed_sequence(memory_bank)[0]
        return memory_bank, hidden_final


class PointerNetRNNDecoder(nn.Module):
    rnn: nn.RNNBase
    attention: Attention

    def __init__(self, rnn_type: str, bidirectional: bool, num_layers: int, input_size: int, hidden_size: int,
                 dropout: float):
        super(PointerNetRNNDecoder, self).__init__()
        if bidirectional:
            assert hidden_size % 2 == 0
            rnn_hidden_size = hidden_size // 2
        else:
            rnn_hidden_size = hidden_size
        self.rnn = rnn_init(rnn_type, input_size=input_size, hidden_size=rnn_hidden_size, bidirectional=bidirectional,
                            num_layers=num_layers, dropout=dropout)
        self.attention = Attention(hidden_size)

    def forward(self, target: Tensor, memory_bank: Tensor, hidden: Tuple[Tensor],
                memory_lengths: Tensor = None) -> Tensor:
        rnn_output, hidden_final = self.rnn(target, hidden)
        memory_bank = memory_bank.transpose(0, 1)
        rnn_output = rnn_output.transpose(0, 1)
        attn_h, align_score = self.attention(memory_bank, rnn_output, memory_lengths)
        return align_score


class PointerNet(nn.Module):
    encoder: RNNEncoder
    decoder: PointerNetRNNDecoder

    def __init__(self, rnn_type: str, bidirectional: bool, num_layers: int, encoder_input_size: int,
                 rnn_hidden_size: int, dropout: float):
        """
        Pointer network
        Args:
            rnn_type: rnn cell type
            bidirectional: whether rnn is bidirectional
            num_layers: number of layers of stacked rnn
            encoder_input_size: input size of encoder
            rnn_hidden_size: rnn hidden dimension size
            dropout: dropout rate
        """
        super(PointerNet, self).__init__()
        self.encoder = RNNEncoder(rnn_type, bidirectional, num_layers, encoder_input_size, rnn_hidden_size, dropout)
        self.decoder = PointerNetRNNDecoder(rnn_type, bidirectional, num_layers, encoder_input_size, rnn_hidden_size,
                                            dropout)

    def forward(self, input: Tensor, input_len: Tensor, output: Tensor, output_len: Tensor) -> Tensor:
        input = input.transpose(0, 1)
        output = output.transpose(0, 1)
        memory_bank, hidden_final = self.encoder(input, input_len)
        align_score = self.decoder(output, memory_bank, hidden_final, input_len)
        return align_score


class PointerNetLoss(nn.Module):

    def __init__(self):
        super(PointerNetLoss, self).__init__()

    def forward(self, target: Tensor, logits: Tensor, lengths: Tensor) -> Tensor:
        """

        Args:
            target: labelled data (batch_size, target_max_len)
            logits: predicts (batch_size, target_max_len, src_max_len)
            lengths: length of label data (batch_size)

        Returns:

        """
        _, target_max_len = target.size()
        logits_flat = logits.view(-1, logits.size(-1))
        log_logits_flat = torch.log(logits_flat)
        target = target.long()
        target_flat = target.view(-1, 1)
        losses_flat = -torch.gather(log_logits_flat, dim=1, index=target_flat)
        losses = losses_flat.view(*target.size())
        mask = sequence_mask(lengths, target_max_len)
        mask = Variable(mask)
        losses = losses * mask.float()
        loss = losses.sum() / lengths.float().sum()
        return loss
