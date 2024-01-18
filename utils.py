import os
import numpy as np
import torch
import scipy.sparse as sp

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

#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_h = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for h in range(num_entities):
        if (h, target_r, target_t) not in triplets_to_filter:
            filtered_h.append(h)
    return torch.LongTensor(filtered_h)

def filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_t = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for t in range(num_entities):
        if (target_h, target_r, t) not in triplets_to_filter:
            filtered_t.append(t)
    return torch.LongTensor(filtered_t)

def get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity):
    """ Perturb object in the triplets
    """
    num_entities = num_entity
    ranks = []

    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]
        # print('t',target_t)
        if entity == 'object':
            filtered_t = filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_t_idx = int((filtered_t == target_t).nonzero())
            _, indices = torch.sort(score[idx][filtered_t], descending=True)
            rank = int((indices == target_t_idx).nonzero())
        if entity == 'subject':
            filtered_h = filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_h_idx = int((filtered_h == target_h).nonzero())
            _, indices = torch.sort(score[idx][filtered_h], descending=True)
            rank = int((indices == target_h_idx).nonzero())

        ranks.append(rank)
    return torch.LongTensor(ranks)


def calc_filtered_test_mrr(num_entity, score, train_triplets, valid_triplets, valid_triplets2, test_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])
        valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])
        valid_triplets2 = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets2])
        test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets])

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, valid_triplets2, test_triplets]).tolist()

        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        ranks = get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

#######################################################################
#
# Utility functions for evaluations (time-aware-filtered)
#
#######################################################################

def ta_filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_h = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for h in range(num_entities):
        if (h, target_r, target_t) not in triplets_to_filter:
            filtered_h.append(h)
    return torch.LongTensor(filtered_h)

def ta_filter_t(triplets_to_filter, target_h, target_r, target_t, target_tim, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    target_tim = int(target_tim)
    filtered_t = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t, target_tim) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t, target_tim))
    # Do not consider an object if it is part of a triplet to filter
    for t in range(num_entities):
        if (target_h, target_r, t, target_tim) not in triplets_to_filter:
            filtered_t.append(t)
    return torch.LongTensor(filtered_t)

def ta_get_filtered_rank(num_entity, score, h, r, t, tim, test_size, triplets_to_filter, entity):
    """ Perturb object in the triplets
    """
    num_entities = num_entity
    ranks = []

    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]
        target_tim = tim[idx]
        # print('t',target_t)
        if entity == 'object':
            filtered_t = ta_filter_t(triplets_to_filter, target_h, target_r, target_t, target_tim, num_entities)
            target_t_idx = int((filtered_t == target_t).nonzero())
            _, indices = torch.sort(score[idx][filtered_t], descending=True)
            rank = int((indices == target_t_idx).nonzero())
        if entity == 'subject':
            filtered_h = ta_filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_h_idx = int((filtered_h == target_h).nonzero())
            _, indices = torch.sort(score[idx][filtered_h], descending=True)
            rank = int((indices == target_h_idx).nonzero())

        ranks.append(rank)
    return torch.LongTensor(ranks)


def ta_calc_filtered_test_mrr(num_entity, score, valid_triplets2, test_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        tim = test_triplets[:, 3]
        test_size = test_triplets.shape[0]

        valid_triplets2 = torch.Tensor([[quad[0], quad[1], quad[2], quad[3]] for quad in valid_triplets2]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in valid_triplets2}

        ranks = ta_get_filtered_rank(num_entity, score, h, r, t, tim, test_size, triplets_to_filter, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

