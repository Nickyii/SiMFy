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
import tracemalloc
import gc


from torch import cuda
device = 'cuda:1' if cuda.is_available() else 'cpu'


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])

def load_quadruples(inPath, fileName, num_r):
    quadrupleList = []
    times = set()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([tail, rel + num_r, head, time])
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)

def load_list(inPath, entityDictPath, relationDictPath):
    entity_list = []
    relation_list = []
    with open(os.path.join(inPath, entityDictPath), 'r') as fr:
        for line in fr:
            line_split = line.split()
            # id = int(line_split[-1])
            text = line_split[0]
            if len(line_split) > 2:
                for i in line_split[1:-1]:
                    text += " " + i
            entity_list.append(text)
    with open(os.path.join(inPath, relationDictPath), 'r') as fr:
        for line in fr:
            line_split = line.split()

            # id = int(line_split[-1])

            text = line_split[0]
            if len(line_split) > 2:
                for i in line_split[1:-1]:
                    text += " " + i
            relation_list.append(text)

    return entity_list, relation_list

def get_outputs(dataset, s_list, p_list, t_list, num_rels, k, is_multi_step=False):
    outputs = []
    if not is_multi_step:
        freq_graph = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset))
        for idx in range(len(s_list)):
            s = s_list[idx]
            p = p_list[idx]
            num_r_2 = num_rels * 2
            row = s * num_r_2 + p
            outputs.append(freq_graph[row].toarray()[0] * k)
    else:
        unique_t_list = list(set(t_list))
        tim_seq_dict = {}
        for tim in unique_t_list:
            tim_seq_dict[str(tim)] = sp.load_npz(
                './data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, tim))
        for idx in range(len(s_list)):
            s = s_list[idx]
            p = p_list[idx]
            t = t_list[idx]
            num_r_2 = num_rels * 2
            row = s * num_r_2 + p
            outputs.append(tim_seq_dict[str(t)][row].toarray()[0] * k)

    return torch.tensor(outputs)

def get_freq(dataset, s_list, p_list, t_list, num_rels):
    outputs = []
    unique_t_list = list(set(t_list))
    tim_seq_dict = {}
    for tim in unique_t_list:
        tim_seq_dict[str(tim)] = sp.load_npz(
            './data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, tim))
    for idx in range(len(s_list)):
        s = s_list[idx]
        p = p_list[idx]
        t = t_list[idx]
        num_r_2 = num_rels * 2
        row = s * num_r_2 + p
        outputs.append(tim_seq_dict[str(t)][row].toarray()[0])

    return torch.tensor(outputs)

class TestDataset(Dataset):

    def __init__(self, quadrupleList):
        self.data = quadrupleList
        self.targets = self.get_targets()
        self.times = self.get_times()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        quad = self.data[index]
        target = self.targets[index]
        tim = self.times[index]
        return {
            'quad': torch.tensor(quad, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long),
            't': torch.tensor(tim, dtype=torch.long),
            's': torch.tensor(quad[0], dtype=torch.long),
            'p': torch.tensor(quad[1], dtype=torch.long),
            'o': torch.tensor(quad[2], dtype=torch.long)
        }

    def get_targets(self):
        targets = []
        for quad in self.data:
            targets.append(quad[2])
        return targets

    def get_times(self):
        times = []
        for quad in self.data:
            times.append(quad[3])
        return times

class TestMLP(nn.Module):
    def __init__(self, num_e, num_rel, num_t, embedding_dim):
        super(TestMLP, self).__init__()

        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel

        # entity relation embedding
        self.rel_embeds = nn.Parameter(torch.zeros(2 * self.num_rel, embedding_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.linear_pred_layer_s1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * embedding_dim, embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * embedding_dim, embedding_dim)

        self.weights_init(self.linear_pred_layer_s1)
        self.weights_init(self.linear_pred_layer_o1)
        self.weights_init(self.linear_pred_layer_s2)
        self.weights_init(self.linear_pred_layer_o2)


        self.dropout = nn.Dropout(0.5)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, s, p, o, freq):
        nce_loss, preds = self.calculate_nce_loss(s, p, o, self.rel_embeds, self.linear_pred_layer_s1, freq)
        return nce_loss, preds

    def calculate_nce_loss(self, s, p, o, rel_embeds, linear1, freq):
        preds_raw1 = self.tanh(linear1(self.dropout(torch.cat((self.entity_embeds[s], rel_embeds[p]), dim=1))))
        preds1 = preds_raw1.mm(self.entity_embeds.transpose(0, 1))

        # encoded_mask = torch.Tensor(np.array(freq.cpu() == 0, dtype=float) * (-100))
        # encoded_mask = encoded_mask.to(device)
        #
        # preds_freq = F.softmax(preds1 + encoded_mask, dim=1)

        preds_freq = F.softmax(preds1 + freq, dim=1)

        nce = torch.sum(torch.gather(torch.log(preds_freq), 1, o.view(-1, 1)))
        nce /= -1. * o.shape[0]

        pred_actor2 = torch.argmax(preds_freq, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, o))
        accuracy = 1. * correct.item() / o.shape[0]
        print('# Batch accuracy', accuracy)

        return nce, preds_freq


def train(dataset):

    embedding_dim = 200
    learning_rate = 1e-3
    weight_decay = 1e-5
    grad_norm = 1.0
    max_epochs = 30

    settings = {}
    settings["embedding_dim"] = embedding_dim
    settings["learning_rate"] = learning_rate
    settings["weight_decay"] = weight_decay
    settings["grad_norm"] = grad_norm
    settings["max_epochs"] = max_epochs

    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    train_data, times = load_quadruples('./data/' + dataset, 'train.txt', num_rels)
    dev_data, dev_times = load_quadruples('./data/' + dataset, 'valid.txt', num_rels)

    # train_data = np.concatenate((train_data, dev_data), axis=0)

    train_set = TestDataset(train_data)

    train_params = {'batch_size': 1024,
                   'shuffle': False,
                   'num_workers': 0
                   }

    train_loader = DataLoader(train_set, **train_params)

    model = TestMLP(num_nodes, num_rels, num_t, embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.to(device)
    now = datetime.datetime.now()
    dt_string = dataset
    main_dirName = os.path.join("TEST_RE/no_fix_freq_models/" + dt_string)
    if not os.path.exists(main_dirName):
        os.makedirs(main_dirName)

    model_path = os.path.join(main_dirName, 'models')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    settings['main_dirName'] = main_dirName
    file_training = open(os.path.join(main_dirName, "training_record.txt"), "w")
    # file_training = open(os.path.join(main_dirName, "tv_training_record.txt"), "w")
    file_training.write("Training Configuration: \n")
    for key in settings:
        file_training.write(key + ': ' + str(settings[key]) + '\n')
    print("Start training...")
    file_training.write("Training Start \n")
    file_training.write("===============================\n")

    for epoch in range(max_epochs):
        model.train()
        print('$Start Epoch: ', epoch)
        loss_epoch = 0
        time_begin = time.time()
        _batch = 0

        for _, data in enumerate(train_loader, 0):

            s_list = data['s'].to(device, dtype=torch.long)
            p_list = data['p'].to(device, dtype=torch.long)
            o_list = data['o'].to(device, dtype=torch.long)
            t_list = data['t'].to(device, dtype=torch.long)

            freq = get_freq(
                dataset, s_list.cpu().detach().numpy().tolist(), p_list.cpu().detach().numpy().tolist(),
                t_list.cpu().detach().numpy().tolist(), num_rels)

            # torch.softmax(freq.float(), dim=1)

            freq = freq.to(device)

            batch_loss, preds = model(s_list, p_list, o_list, freq)
            if batch_loss is not None:
                error = batch_loss
            else:
                continue
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += error.item()

            print("batch: " + str(_) + ' finished. Used time: ' +
                  str(time.time() - time_begin) + ', loss: ' + str(error.item()))
            file_training.write(
                "epoch: " + str(epoch) + "batch: " + str(_) + ' finished. Used time: '
                + str(time.time() - time_begin) + ', Loss: ' + str(error.item()) + '\n')
            _batch += 1

        epoch_time = time.time()
        print("Done\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
              format(epoch, loss_epoch / _batch, epoch_time - time_begin))
        file_training.write("******\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
                            format(epoch, loss_epoch / _batch, epoch_time - time_begin) + '\n')


    torch.save(model, model_path + '/' + dataset + '_best.pth')
    # torch.save(model, model_path + '/' + dataset + '_tv_best.pth')
    print("Training done")
    file_training.write("Training done")
    file_training.close()



def test(dataset):

    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    test_data, times = load_quadruples('./data/' + dataset, 'test.txt', num_rels)

    dt_string = dataset
    main_dirName = os.path.join("TEST_RE/no_fix_freq_models/" + dt_string)
    model_path = os.path.join(main_dirName, 'models')
    file_testing = open(os.path.join(main_dirName, "testing_record.txt"), "w")
    file_testing.write("Testing Start \n")
    file_testing.write("===============================\n")

    test_set = TestDataset(test_data)

    test_params = {'batch_size': 1024,
                   'shuffle': False,
                   'num_workers': 0
                   }

    test_loader = DataLoader(test_set, **test_params)

    model = torch.load(model_path + '/' + dataset + '_best.pth')

    model.to(device)

    model.eval()

    with torch.no_grad():

        mrr, hit1, hit3, hit10 = 0, 0, 0, 0

        for _, data in tqdm(enumerate(test_loader, 0)):

            targets = data['target'].to(device, dtype=torch.long)
            s_list = data['s'].to(device, dtype=torch.long)
            p_list = data['p'].to(device, dtype=torch.long)
            o_list = data['o'].to(device, dtype=torch.long)
            t_list = data['t'].to(device, dtype=torch.long)

            freq = get_freq(
                dataset, s_list.cpu().detach().numpy().tolist(), p_list.cpu().detach().numpy().tolist(),
                t_list.cpu().detach().numpy().tolist(), num_rels)

            # torch.softmax(freq.float(), dim=1)

            freq = freq.to(device)

            l, outputs = model(s_list, p_list, o_list, freq)

            batch_mrr, batch_hit1, batch_hit3, batch_hit10 = calc_raw_mrr(outputs, targets, hits=[1, 3, 10])

            mrr += batch_mrr * len(targets)
            hit1 += batch_hit1 * len(targets)
            hit3 += batch_hit3 * len(targets)
            hit10 += batch_hit10 * len(targets)

        mrr = mrr / test_data.shape[0]
        hit1 = hit1 / test_data.shape[0]
        hit3 = hit3 / test_data.shape[0]
        hit10 = hit10 / test_data.shape[0]

        size = len(test_data)
        print(f"|S|: {size}")
        file_testing.write(f"|S|: {size}" + '\n')

        print(f"MRR: {mrr}")
        file_testing.write(f"MRR: {mrr}" + '\n')

        print(f"Hits@1: {hit1}")
        print(f"Hits@3: {hit3}")
        print(f"Hits@10: {hit10}")

        file_testing.write(f"Hits@1: {hit1}" + '\n')
        file_testing.write(f"Hits@3: {hit3}" + '\n')
        file_testing.write(f"Hits@10: {hit10}" + '\n')

        file_testing.write("===============================\n")
        file_testing.write("Testing done")
        file_testing.close()


def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(score, labels, hits=[]):
    with torch.no_grad():

        ranks = sort_and_rank(score, labels)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()


if __name__ == '__main__':

    dataset = 'ICEWS05-15'
    train(dataset)

    # for dataset in ['ICEWS05-15', 'ICEWS18']:
    #     train(dataset)
    #     test(dataset)

    # dataset = 'testshort'
    test(dataset)

    # for dataset in ['GDELT', 'WIKI', 'YAGO']:
    #     test(dataset)