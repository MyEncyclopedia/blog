# coding=utf-8
from typing import Tuple, List
import numpy as np
from torch.utils.data import Dataset
from copy import copy


class TSPDataset(Dataset):
    "each data item of form (input, input_len, output_in, output_out, output_len)"
    data: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

    def __init__(self, filename: str, max_in_seq_len: int, max_out_seq_len: int):
        """
        TSP dataset
        Args:
            filename: dataset file name
            max_in_seq_len: maximum input sequence length
            max_out_seq_len: maximum output sequence length
        """

        super(TSPDataset, self).__init__()
        self.max_in_seq_len = max_in_seq_len
        self.max_out_seq_len = max_out_seq_len
        self.START = [0, 0]
        self.END = [0, 0]
        self.load_data(filename)

    def load_data(self, filename: str):
        with open(filename, 'r') as f:
            data = []
            for line in f:
                input_raw, output_raw = line.strip().split('output')
                input_raw = list(map(float, input_raw.strip().split(' ')))
                output_raw = list(map(int, output_raw.strip().split(' ')))
                # Add START token
                output_in = copy(self.START)
                output_out = []
                for idx in output_raw:
                    output_in += input_raw[2 * (idx - 1): 2 * idx]
                    output_out += [idx]
                # Add END token
                output_out += [0]

                # Padding input
                input_len = len(input_raw) // 2
                input = self.START + input_raw
                input_len += 1
                # Special START token
                assert self.max_in_seq_len + 1 >= input_len
                for i in range(self.max_in_seq_len + 1 - input_len):
                    input += self.END
                input = np.array(input).reshape([-1, 2])  # shape = (1 + num_points, 2)
                input_len = np.array([input_len])         # 1 + num_points
                # Padding output
                output_len = len(output_raw) + 1
                for i in range(self.max_out_seq_len + 1 - output_len):
                    output_in += self.START
                output_in = np.array(output_in).reshape([-1, 2]) # shape = (max_out_seq_len + 1, 2)
                output_out = output_out + [0] * (self.max_out_seq_len + 1 - output_len)
                output_out = np.array(output_out)  # shape = (max_out_seq_len + 1)
                output_len = np.array([output_len])  # length_output + 1

                data.append((input.astype("float32"), input_len, output_in.astype("float32"), output_out, output_len))
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, input_len, output_in, output_out, output_len = self.data[index]
        return input, input_len, output_in, output_out, output_len
