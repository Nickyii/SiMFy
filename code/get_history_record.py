import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def get_freq(dataset):
    num_e, num_r = get_total_number('./data/{}'.format(dataset), 'stat.txt')
    train_data, train_times = load_quadruples('./data/{}'.format(dataset), 'train.txt', num_r)  
    dev_data, dev_times = load_quadruples('./data/{}'.format(dataset), 'valid.txt', num_r)  
    test_data, test_times = load_quadruples('./data/{}'.format(dataset), 'test.txt', num_r) 
    all_data = np.concatenate((train_data, dev_data, test_data), axis=0) 
    all_times = np.concatenate((train_times, dev_times, test_times))
    

    save_dir_obj = './data/{}/history_seq/'.format(dataset)

    mkdirs(save_dir_obj)

    
    num_r_2 = num_r * 2
    row = all_data[:, 0] * num_r_2 + all_data[:, 1]  
    col_rel = all_data[:, 1]
    d_ = np.ones(len(row))
    tail_rel = sp.csr_matrix((d_, (row, col_rel)), shape=(num_e * num_r_2, num_r_2))  
    sp.save_npz('./data/{}/history_seq/h_r_seq_rel.npz'.format(dataset), tail_rel)

    
    all_data_his = np.concatenate((train_data, dev_data), axis=0) 
    num_r_2 = num_r * 2
    row = all_data_his[:, 0] * num_r_2 + all_data_his[:, 1]  
    col = all_data_his[:, 2]
    d_ = np.ones(len(row))
    tail_all = sp.csr_matrix((d_, (row, col)), shape=(num_e * num_r_2, num_e))
    sp.save_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset), tail_all)

    
    num_r_2 = num_r * 2
    row = train_data[:, 0] * num_r_2 + train_data[:, 1]
    col = train_data[:, 2]
    d_ = np.ones(len(row))
    tail_train = sp.csr_matrix((d_, (row, col)), shape=(num_e * num_r_2, num_e))
    sp.save_npz('./data/{}/history_seq/h_r_history_train.npz'.format(dataset), tail_train)

    for idx, tim in tqdm(enumerate(all_times)):  
        
        test_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if quad[3] == tim])
        # get object_entities
        row = test_new_data[:, 0] * num_r_2 + test_new_data[:, 1]  
        col = test_new_data[:, 2]
        d = np.ones(len(row))
        tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r_2, num_e)) 
        sp.save_npz('./data/{}/history_seq/h_r_history_seq_{}.npz'.format(dataset, tim), tail_seq)

def get_multi_freq(dataset, mode="test"):
    num_e, num_r = get_total_number('./data/{}'.format(dataset), 'stat.txt')
    num_r_2 = num_r * 2

    if mode == "test":
        test_data, test_times = load_quadruples('./data/{}'.format(dataset), 'test.txt', num_r) 
        start_tim = test_times[0]

        train_valid_tail_seq = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset))
        sp.save_npz('./data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, start_tim),
                    train_valid_tail_seq)

        for idx, tim in tqdm(enumerate(test_times)):  
            if idx != 0:
                new_seq = sp.load_npz(
                    './data/{}/history_seq/h_r_history_seq_{}.npz'.format(dataset, test_times[idx - 1]))
                old_seq = sp.load_npz(
                    './data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, test_times[idx - 1]))

                old_seq += new_seq
                sp.save_npz('./data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, tim), old_seq)

    elif mode == "valid":
        pass
    elif mode == "train":
        train_data, train_times = load_quadruples('./data/{}'.format(dataset), 'train.txt', num_r)
        start_tim = train_times[1]

        first_seq = sp.load_npz(
            './data/{}/history_seq/h_r_history_seq_{}.npz'.format(dataset, train_times[0]))
        sp.save_npz('./data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, start_tim),
                    first_seq)

        init_seq = sp.csr_matrix((num_e * num_r_2, num_e))
        sp.save_npz('./data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, train_times[0]), init_seq)
        # print(init_seq.toarray())

        for idx, tim in tqdm(enumerate(train_times)): 
            if idx >= 2:
                new_seq = sp.load_npz(
                    './data/{}/history_seq/h_r_history_seq_{}.npz'.format(dataset, train_times[idx - 1]))
                old_seq = sp.load_npz(
                    './data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, train_times[idx - 1]))

                old_seq += new_seq
                sp.save_npz('./data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, tim), old_seq)


if __name__ == '__main__':

    # for dataset in ['test', 'testshort', 'GDELT', 'ICEWS05-15', 'ICEWS14', 'ICEWS18', 'WIKI', 'YAGO']:
    #     get_freq(dataset)

    # dataset = "ICEWS14"

    for dataset in ['testshort', 'test', 'ICEWS05-15', 'ICEWS14', 'ICEWS18']:
        get_multi_freq(dataset, mode="train")

