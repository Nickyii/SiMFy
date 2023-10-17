import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import random
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from torch import cuda
device = 'cuda:3' if cuda.is_available() else 'cpu'


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

# def get_outputs(dataset, s_list, p_list, t_list, times, num_nodes, num_rels, train_valid_tail_seq, k, is_multi_step=False):
#     outputs = []
#     freq_graph = train_valid_tail_seq
#     if not is_multi_step:
#         for idx in range(len(s_list)):
#             s = s_list[idx]
#             p = p_list[idx]
#             num_r_2 = num_rels * 2
#             row = s * num_r_2 + p
#             outputs.append(freq_graph[row].toarray()[0] * k)
#     else:
#         cur_tim = t_list[0]
#         start_tim = times[0]
#         for idx in range(len(s_list)):
#             s = s_list[idx]
#             p = p_list[idx]
#             t = t_list[idx]
#             if t != cur_tim:
#                 if t > cur_tim:
#                     new_tim_tail_seq = sp.load_npz('./data/{}/history_seq/h_r_history_seq_{}.npz'.format(dataset, cur_tim))
#                     freq_graph += new_tim_tail_seq
#                     cur_tim = t
#                 else:
#                     freq_graph = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset)) # 初始图
#                     cur_tim = t
#             num_r_2 = num_rels * 2
#             row = s * num_r_2 + p
#             outputs.append(freq_graph[row].toarray()[0] * k)
#
#     return torch.tensor(outputs), freq_graph

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


def test(dataset, multi_step=False, equal=True, hard=True):

    # train_valid_tail_seq = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset))
    # train_tail_seq = sp.load_npz('./data/{}/history_seq/h_r_history_train.npz'.format(dataset))

    k = 2 

    if dataset in ['test', 'testshort', 'ICEWS05-15', 'ICEWS14', 'ICEWS18']:
        entity_list, relation_list = load_list("./data/" + dataset, "entity2id.txt", "relation2id.txt")

    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    test_data, times = load_quadruples('./data/' + dataset, 'test.txt', num_rels)

    is_multi = "single"
    if multi_step:
        is_multi = "multi"

    file_testing = open(os.path.join("TEST_RE", str(k) + "_" + dataset + "_" + is_multi + "_testing_record.txt"), "w")
    file_testing.write("Testing Start \n")
    file_testing.write("===============================\n")

    test_set = TestDataset(test_data)

    test_params = {'batch_size': 1024,
                   'shuffle': False,
                   'num_workers': 0
                   }

    test_loader = DataLoader(test_set, **test_params)

    fin_targets = []
    idx_re = []




    with torch.no_grad():

        for _, data in tqdm(enumerate(test_loader, 0)):

            target_list = data['target'].numpy().tolist()
            s_list = data['s'].numpy().tolist()
            p_list = data['p'].numpy().tolist()
            o_list = data['o'].numpy().tolist()
            t_list = data['t'].numpy().tolist()

            outputs = get_outputs(dataset, s_list, p_list, t_list, num_rels, k, multi_step)
            # target_list = targets.cpu().detach().numpy().tolist()
            fin_targets.extend(target_list)
            res = torch.softmax(outputs.float(), dim=1).numpy().tolist()

            for i, re in enumerate(res):
                idx = sorted(range(len(re)), key=lambda k: -re[k])
                idx_re.append(idx)

                # s = s_list[i]
                # p = p_list[i]
                # if p > num_rels - 1:
                #     p = p - num_rels
                # o = target_list[i]
                # print(f"ground truth: {o}")
                # file_testing.write(f"ground truth: {o}" + '\n')
                # rank_list = idx[:10]
                # print(f"rank list: {rank_list}")
                # file_testing.write(f"rank list: {rank_list}" + '\n')
                #
                # print(f"o: {entity_list[o]}  (s,p,o): ({entity_list[s]} , {relation_list[p]} , {entity_list[o]})")
                # file_testing.write(
                #     f"o: {entity_list[o]}  (s,p,o): ({entity_list[s]} , {relation_list[p]} , {entity_list[o]})" + '\n')
                #
                # print('o'.ljust(30)+'score'+'     rank')
                # file_testing.write('o'.ljust(30)+'score'+'     rank' + '\n')
                # for _, pred_o in enumerate(rank_list, 0):
                #     print(f"{entity_list[pred_o].ljust(30)}{re[pred_o]:.3f}     {_+1}")
                #     file_testing.write(f"{entity_list[pred_o].ljust(30)}{re[pred_o]:.3f}     {_+1}" + '\n')
                # print("\n")
                # file_testing.write('\n')

            if _ % 2 == 0 and dataset in ['test', 'testshort',  'ICEWS05-15', 'ICEWS14', 'ICEWS18']:
                re = res[0]
                s = s_list[0]
                p = p_list[0]
                if p > num_rels - 1:
                    p = p - num_rels
                o = target_list[0]
                print(f"ground truth: {o}")
                file_testing.write(f"ground truth: {o}" + '\n')
                idx = sorted(range(len(re)), key=lambda k: -re[k])
                rank_list = idx[:10]
                print(f"rank list: {rank_list}")
                file_testing.write(f"rank list: {rank_list}" + '\n')

                print(f"o: {entity_list[o]}  (s,p,o): ({entity_list[s]} , {relation_list[p]} , {entity_list[o]})")
                file_testing.write(
                    f"o: {entity_list[o]}  (s,p,o): ({entity_list[s]} , {relation_list[p]} , {entity_list[o]})" + '\n')

                print('o'.ljust(30)+'score'+'     rank')
                file_testing.write('o'.ljust(30)+'score'+'     rank' + '\n')
                for _, pred_o in enumerate(rank_list, 0):
                    print(f"{entity_list[pred_o].ljust(30)}{re[pred_o]:.3f}     {_+1}")
                    file_testing.write(f"{entity_list[pred_o].ljust(30)}{re[pred_o]:.3f}     {_+1}" + '\n')
                print("\n")
                file_testing.write('\n')

    # print(fin_targets)
    # print(idx_re)

    rank = []
    is_hit1 = []
    is_hit3 = []
    is_hit10 = []

    for i, target in tqdm(enumerate(fin_targets, 0)):
        r = idx_re[i].index(target) + 1
        rank.append(r)
        if r == 1:
            h1 = 1
            h3 = 1
            h10 = 1
        elif r <= 3:
            h1 = 0
            h3 = 1
            h10 = 1
        elif r <= 10:
            h1 = 0
            h3 = 0
            h10 = 1
        else:
            h1 = 0
            h3 = 0
            h10 = 0
        is_hit1.append(h1)
        is_hit3.append(h3)
        is_hit10.append(h10)

    size = len(fin_targets)
    print(f"|S|: {size}")
    file_testing.write(f"|S|: {size}" + '\n')

    mrr = 0
    s = 0
    for r in rank:
        s += 1.0 / r
    mrr = s / size
    print(f"MRR: {mrr}")
    file_testing.write(f"MRR: {mrr}" + '\n')

    hit1 = sum(is_hit1) / float(size)
    hit3 = sum(is_hit3) / float(size)
    hit10 = sum(is_hit10) / float(size)

    print(f"Hits@1: {hit1}")
    print(f"Hits@3: {hit3}")
    print(f"Hits@10: {hit10}")

    file_testing.write(f"Hits@1: {hit1}" + '\n')
    file_testing.write(f"Hits@3: {hit3}" + '\n')
    file_testing.write(f"Hits@10: {hit10}" + '\n')

    file_testing.write("===============================\n")
    file_testing.write("Testing done")
    file_testing.close()



if __name__ == '__main__':

    # dataset = 'ICEWS18'
    # test(dataset, True)


    for dataset in ['test', 'testshort', 'ICEWS05-15', 'ICEWS14', 'ICEWS18', 'WIKI', 'YAGO']:
        test(dataset, multi_step=True)
