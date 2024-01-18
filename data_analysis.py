import os
import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)


def get_unseen_ratio(args):
    dataset = args.dataset
    multi_step = args.multi_step
    num_e, num_r = get_total_number('./data/{}'.format(dataset), 'stat.txt')
    test_data, times = load_quadruples_origin('./data/' + dataset, 'test.txt')

    is_multi = "single"
    if multi_step:
        is_multi = "multi"

    main_dirName = os.path.join("TEST_RE/analysis/")
    if not os.path.exists(main_dirName):
        os.makedirs(main_dirName)

    file_analysis = open(os.path.join(main_dirName, dataset + "_" + is_multi + "_analysis.txt"), "w")
    file_analysis.write("Analysis Start \n")
    file_analysis.write("===============================\n")
    unseen = 0

    tim_seq_dict = {}

    if multi_step:
        for tim in tqdm(times):
            tim_seq_dict[str(tim)] = sp.load_npz(
                './data/{}/history_seq/all_history_seq_before_{}.npz'.format(dataset, tim))

    freq_graph = sp.load_npz('./data/{}/history_seq/h_r_history_train_valid.npz'.format(dataset))

    for idx, quad in tqdm(enumerate(test_data)):
        if not multi_step:
            s = quad[0]
            p = quad[1]
            o = quad[2]
            num_r_2 = num_r * 2
            row = s * num_r_2 + p
            if freq_graph[row, o] == 0:
                unseen += 1
        else:
            s = quad[0]
            p = quad[1]
            o = quad[2]
            t = quad[3]
            num_r_2 = num_r * 2
            row = s * num_r_2 + p
            if tim_seq_dict[str(t)][row, o] == 0:
                unseen += 1

    size = len(test_data)
    print(f"|S|: {size}")
    file_analysis.write(f"|S|: {size}" + '\n')

    print(f"unseen events: {unseen}")
    file_analysis.write(f"unseen events: {unseen}" + '\n')

    print(f"repetitive events: {size - unseen}")
    file_analysis.write(f"repetitive events: {size - unseen}" + '\n')

    print(f"unseen ratio: {int(unseen/size * 100)}%")
    file_analysis.write(f"unseen ratio: {int(unseen/size * 100)}%" + '\n')

    file_analysis.write("===============================\n")
    file_analysis.write("Analysis done")
    file_analysis.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analysis')
    parser.add_argument("--multi_step", action='store_true', help="whether to multi-step analyze")
    parser.add_argument("-d", "--dataset", type=str, default='ICEWS14', help="dataset")

    args = parser.parse_args()
    get_unseen_ratio(args)

