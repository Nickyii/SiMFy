import argparse
import datetime
import time
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from simfy_model import SimfyDataset, Simfy
from test import test

from torch import cuda


def train(args):

    dataset = args.dataset
    embedding_dim = args.embedding_dim
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    grad_norm = args.grad_norm
    max_epochs = args.max_epochs

    device = 'cuda:' + str(args.gpu) if cuda.is_available() else 'cpu'

    settings = {}
    settings["embedding_dim"] = embedding_dim
    settings["learning_rate"] = learning_rate
    settings["weight_decay"] = weight_decay
    settings["grad_norm"] = grad_norm
    settings["max_epochs"] = max_epochs

    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    train_data, times = load_quadruples('./data/' + dataset, 'train.txt', num_rels)

    train_set = SimfyDataset(train_data)

    train_params = {'batch_size': 1024,
                   'shuffle': False,
                   'num_workers': 0
                   }

    train_loader = DataLoader(train_set, **train_params)

    model = Simfy(num_nodes, num_rels, num_t, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    now = datetime.datetime.now()
    dt_string = dataset
    main_dirName = os.path.join("TEST_RE/models/" + dt_string)
    if not os.path.exists(main_dirName):
        os.makedirs(main_dirName)

    model_path = os.path.join(main_dirName, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    settings['main_dirName'] = main_dirName
    file_training = open(os.path.join(main_dirName, "training_record.txt"), "w")
    file_training.write("Training Configuration: \n")
    for key in settings:
        file_training.write(key + ': ' + str(settings[key]) + '\n')
    print("######## Training Start ########")
    file_training.write("Training Start \n")
    file_training.write("===============================\n")

    for epoch in tqdm(range(max_epochs)):
        model.train()
        loss_epoch = 0
        time_begin = time.time()
        _batch = 0

        for _, data in enumerate(train_loader, 0):

            s_list = data['s'].to(device, dtype=torch.long)
            p_list = data['p'].to(device, dtype=torch.long)
            o_list = data['o'].to(device, dtype=torch.long)
            t_list = data['t'].to(device, dtype=torch.long)

            batch_loss, preds = model(s_list, p_list, o_list)
            if batch_loss is not None:
                error = batch_loss
            else:
                continue
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += error.item()

            file_training.write(
                "epoch: " + str(epoch) + "batch: " + str(_) + ' finished. Used time: '
                + str(time.time() - time_begin) + ', Loss: ' + str(error.item()) + '\n')
            _batch += 1

        epoch_time = time.time()
        file_training.write("******\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
                            format(epoch, loss_epoch / _batch, epoch_time - time_begin) + '\n')


    torch.save(model, model_path + '/' + dataset + '_best.pth')
    print("######## Training done ########")
    file_training.write("Training done")
    file_training.close()

def train_simfy(args):
    if args.test:
        test(args)
    else:
        train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simfy')
    parser.add_argument("--dataset", type=str, default='ICEWS14', help="dataset")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--embedding_dim", type=int, default=200, help="embedding dim of entities and relations")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--grad_norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--max_epochs", type=int, default=30, help="epochs to train")

    parser.add_argument("--test", action='store_true', help="to evaluate the model")
    parser.add_argument("--alpha", type=float, default=0.001, help="balance freq and similarity")
    parser.add_argument("--k", type=int, default=2, help="a hyperparameter to balance the extremely small values")
    parser.add_argument("--metric", type=str, default='raw',
                        help="evaluation metrics: choose one in [raw, filtered, time_aware_filtered")
    parser.add_argument("--multi_step", action='store_true', help="whether to multi-step analyze")

    args = parser.parse_args()

    train_simfy(args)

