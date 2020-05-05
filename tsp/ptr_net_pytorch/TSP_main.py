# coding=utf-8

import numpy as np
import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm, clip_grad_norm_
import argparse
import logging
import sys
from tensorboardX import SummaryWriter

from TSP_dataset import TSPDataset
from pointer_net import PointerNet, PointerNetLoss

if __name__ == "__main__":
    # Parse argument
    parser = argparse.ArgumentParser("TSP")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_in_seq_len", type=int, default=20)
    parser.add_argument("--max_out_seq_len", type=int, default=30)
    parser.add_argument("--rnn_hidden_size", type=int, default=128)
    parser.add_argument("--attention_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=5.)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--train_filename", type=str, default="/Users/frank/Desktop/jikecloud/PTR_NETS/tsp_20_test.txt")
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./log")

    args = parser.parse_args()

    # Pytroch configuration
    if args.gpu >= 0 and torch.cuda.is_available():
        args.use_cuda = True
        torch.cuda.device(args.gpu)
    else:
        args.use_cuda = False

    # Logger
    logger = logging.getLogger("TSP")
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = formatter
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG)

    # Summary writer
    writer = SummaryWriter(args.log_dir)

    # Loading data
    train_ds = TSPDataset(args.train_filename, args.max_in_seq_len, args.max_out_seq_len)
    logger.info("Train data size: {}".format(len(train_ds)))

    train_dl = DataLoader(train_ds, num_workers=2, batch_size=args.batch_size)

    # Init model
    model: PointerNet = PointerNet("LSTM", True, args.num_layers, 2, args.rnn_hidden_size, 0.0)
    pointer_net_loss: PointerNetLoss = PointerNetLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.use_cuda:
        model.cuda()

    for epoch in range(args.num_epoch):
        model.train()
        total_loss = 0.
        batch_cnt = 0.
        for batch_input, batch_input_len, batch_output_in, batch_output_out, batch_output_len in train_dl:
            batch_input_v = Variable(batch_input)
            batch_output_in_v = Variable(batch_output_in)
            batch_output_out_v = Variable(batch_output_out)
            if args.use_cuda:
                batch_input_v = batch_input_v.cuda()
                batch_input_len = batch_input_len.cuda()
                batch_output_in_v = batch_output_in_v.cuda()
                batch_output_out_v = batch_output_out_v.cuda()
                batch_output_len = batch_output_len.cuda()

            optimizer.zero_grad()
            align_score = model(batch_input_v, batch_input_len, batch_output_in_v, batch_output_len)
            loss = pointer_net_loss(batch_output_out_v, align_score, batch_output_len)

            l = loss.data.item()
            total_loss += l
            batch_cnt += 1

            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
        writer.add_scalar('train/loss', total_loss / batch_cnt, epoch)
        logger.info("Epoch : {}, loss {}".format(epoch, total_loss / batch_cnt))

        # Checkout
        if epoch % args.checkpoint_interval == args.checkpoint_interval - 1:
            # Save model
            if args.model_file is not None:
                torch.save(model.state_dict(), args.model_file)

