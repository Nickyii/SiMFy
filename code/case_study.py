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

    def forward(self, s, p, o):
        nce_loss, preds = self.calculate_nce_loss(s, p, o, self.rel_embeds, self.linear_pred_layer_s1)
        return nce_loss, preds

    def calculate_nce_loss(self, s, p, o, rel_embeds, linear1):
        preds_raw1 = self.tanh(linear1(self.dropout(torch.cat((self.entity_embeds[s], rel_embeds[p]), dim=1))))
        preds1 = F.softmax(preds_raw1.mm(self.entity_embeds.transpose(0, 1)), dim=1)

        nce = torch.sum(torch.gather(torch.log(preds1), 1, o.view(-1, 1)))
        nce /= -1. * o.shape[0]

        pred_actor2 = torch.argmax(preds1, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, o))
        accuracy = 1. * correct.item() / o.shape[0]
        print('# Batch accuracy', accuracy)

        return nce, preds1


def test(dataset):

    if dataset in ['test', 'testshort', 'ICEWS05-15', 'ICEWS14', 'ICEWS18']:
        entity_list, relation_list = load_list("./data/" + dataset, "entity2id.txt", "relation2id.txt")


    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    test_data, times = load_quadruples('./data/' + dataset, 'test.txt', num_rels)

    freq_graph = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset))


    dt_string = dataset
    main_dirName = os.path.join("TEST_RE/mlp_models/" + dt_string)
    model_path = os.path.join(main_dirName, 'models')
    file_testing = open(os.path.join(main_dirName, "case_study.txt"), "w")
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

        for _, data in tqdm(enumerate(test_loader, 0)):

            targets = data['target'].to(device, dtype=torch.long)
            s_list = data['s'].to(device, dtype=torch.long)
            p_list = data['p'].to(device, dtype=torch.long)
            o_list = data['o'].to(device, dtype=torch.long)
            t_list = data['t'].to(device, dtype=torch.long)

            l, outputs = model(s_list, p_list, o_list)
            target_list = targets.cpu().detach().numpy().tolist()

            res = outputs.cpu().detach().numpy().tolist()


            for i, re in tqdm(enumerate(res)):
                idx = sorted(range(len(re)), key=lambda k: -re[k])

                s = s_list[i]
                p = p_list[i]
                num_r_2 = num_rels * 2
                row = s * num_r_2 + p

                row = row.cpu().detach()

                if p > num_rels - 1:
                    p = p - num_rels
                o = target_list[i]
                rank_list = idx[:10]

                top_rank_list = rank_list[:3]  # top k

                flag = False

                if freq_graph[row, o] == 0 and o in top_rank_list:
                    flag = True


                if flag:

                    print(f"ground truth: {o}, ({s},{p},{o})")
                    file_testing.write(f"ground truth: {o}, ({s},{p},{o})" + '\n')
                    print(f"rank list: {rank_list}")
                    file_testing.write(f"rank list: {rank_list}" + '\n')

                    print(f"o: {entity_list[o]}  (s,p,o): ({entity_list[s]} , {relation_list[p]} , {entity_list[o]})")
                    file_testing.write(
                        f"o: {entity_list[o]}  (s,p,o): ({entity_list[s]} , {relation_list[p]} , {entity_list[o]})" + '\n')

                    print('o'.ljust(30) + 'score' + '     rank')
                    file_testing.write('o'.ljust(30) + 'score' + '     rank' + '\n')
                    for _, pred_o in enumerate(rank_list, 0):
                        print(f"{entity_list[pred_o].ljust(30)}{re[pred_o]:.3f}     {_ + 1}")
                        file_testing.write(f"{entity_list[pred_o].ljust(30)}{re[pred_o]:.3f}     {_ + 1}" + '\n')
                    print("\n")
                    file_testing.write('\n')



    file_testing.write("===============================\n")
    file_testing.write("Testing done")
    file_testing.close()


if __name__ == '__main__':


    # for dataset in ['ICEWS14', 'ICEWS05-15', 'ICEWS18']:
    #     train(dataset)
    #     test(dataset)

    dataset = 'ICEWS05-15'
    test(dataset)

    # for dataset in ['GDELT', 'WIKI', 'YAGO']:
    #     test(dataset)