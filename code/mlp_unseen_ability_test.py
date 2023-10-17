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
device = 'cuda:2' if cuda.is_available() else 'cpu'


def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])

def load_quadruples_origin(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

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
    main_dirName = os.path.join("TEST_RE/mlp_models/" + dt_string)
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

    if dataset in ['test', 'testshort', 'ICEWS05-15', 'ICEWS14', 'ICEWS18']:
        entity_list, relation_list = load_list("./data/" + dataset, "entity2id.txt", "relation2id.txt")

    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    test_data, times = load_quadruples('./data/' + dataset, 'test.txt', num_rels)


    dt_string = dataset
    main_dirName = os.path.join("TEST_RE/mlp_models/" + dt_string)
    model_path = os.path.join(main_dirName, 'models')
    file_testing = open(os.path.join(main_dirName, "testing_record.txt"), "w")
    # file_testing = open(os.path.join(main_dirName, "tv_testing_record.txt"), "w")
    file_testing.write("Testing Start \n")
    file_testing.write("===============================\n")

    test_set = TestDataset(test_data)

    test_params = {'batch_size': 1024,
                   'shuffle': False,
                   'num_workers': 0
                   }

    test_loader = DataLoader(test_set, **test_params)

    model = torch.load(model_path + '/' + dataset + '_best.pth')
    # model = torch.load(model_path + '/' + dataset + '_tv_best.pth')

    model.to(device)

    model.eval()

    fin_targets = []
    idx_re = []


    with torch.no_grad():

        for _, data in tqdm(enumerate(test_loader, 0)):

            targets = data['target'].to(device, dtype=torch.long)
            s_list = data['s'].to(device, dtype=torch.long)
            p_list = data['p'].to(device, dtype=torch.long)
            o_list = data['o'].to(device, dtype=torch.long)
            t_list = data['t'].to(device, dtype=torch.long)

            l, outputs = model(s_list, p_list, o_list)
            target_list = targets.cpu().detach().numpy().tolist()
            fin_targets.extend(target_list)

            res = outputs.cpu().detach().numpy().tolist()

            # tracemalloc.start()
            # current, peak = tracemalloc.get_traced_memory()
            # print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

            for i, re in enumerate(res):
                idx = [sorted(range(len(re)), key=lambda k: -re[k])]
                idx_re.extend(idx)

            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')
            # print("[ Top 10 ]")
            # for stat in top_stats[:10]:
            #     print(stat)

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

            if _ % 2 == 0 and dataset in ['test', 'testshort', 'ICEWS05-15', 'ICEWS14', 'ICEWS18']:
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

    # tracemalloc.stop()
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

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

def test_new(dataset, mode="unseen", multi_step=False):
    is_multi = "single"
    if multi_step:
        is_multi = "multi"

    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    unseen_file_name = os.path.join(is_multi + "_unseen_test.txt")
    re_file_name = os.path.join(is_multi + "_re_test.txt")
    dt_string = dataset
    main_dirName = os.path.join("TEST_RE/mlp_models/" + dt_string)
    model_path = os.path.join(main_dirName, 'models')

    if mode == "unseen":
        test_data, times = load_quadruples('./data/' + dataset, unseen_file_name, num_rels)
        file_testing = open(os.path.join(main_dirName, "unseen_testing_record.txt"), "w")
    elif mode == "re":
        test_data, times = load_quadruples('./data/' + dataset, re_file_name, num_rels)
        file_testing = open(os.path.join(main_dirName, "repetitive_testing_record.txt"), "w")

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

            l, outputs = model(s_list, p_list, o_list)

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

def get_unseen_test_data(dataset, multi_step=False):
    num_e, num_r, _ = get_total_number('./data/{}'.format(dataset), 'stat.txt')
    test_data, times = load_quadruples_origin('./data/' + dataset, 'test.txt')

    is_multi = "single"
    if multi_step:
        is_multi = "multi"

    tim_seq_dict = {}

    if multi_step:
        for tim in tqdm(times):
            tim_seq_dict[str(tim)] = sp.load_npz(
                './data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, tim))

    freq_graph = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset))

    unseen_quadruples = []
    re_quadruples = []

    for idx, quad in tqdm(enumerate(test_data)):
        if not multi_step:
            s = quad[0]
            p = quad[1]
            o = quad[2]
            num_r_2 = num_r * 2
            row = s * num_r_2 + p
            if freq_graph[row, o] == 0:
                unseen_quadruples.append(quad)
            else:
                re_quadruples.append(quad)
        else:
            s = quad[0]
            p = quad[1]
            o = quad[2]
            t = quad[3]
            num_r_2 = num_r * 2
            row = s * num_r_2 + p
            if tim_seq_dict[str(t)][row, o] == 0:
                unseen_quadruples.append(quad)
            else:
                re_quadruples.append(quad)

    main_dirName = os.path.join("./data/", dataset)
    unseen_file_name = os.path.join(main_dirName, is_multi + "_unseen_test.txt")
    re_file_name = os.path.join(main_dirName, is_multi + "_re_test.txt")

    with open(unseen_file_name, 'w') as file:
        for row in unseen_quadruples:
            line = '\t'.join(map(str, row)) + '\n'
            file.write(line)

    with open(re_file_name, 'w') as file:
        for row in re_quadruples:
            line = '\t'.join(map(str, row)) + '\n'
            file.write(line)



if __name__ == '__main__':

    # dataset = 'testshort'
    # test_new(dataset)

    for dataset in ['ICEWS14', 'ICEWS18', 'ICEWS05-15']:
        test_new(dataset)
