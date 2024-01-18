from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils import *
from simfy_model import SimfyDataset, Simfy
from torch import cuda


def test(args):
    dataset = args.dataset
    alpha = args.alpha
    k = args.k
    metric = args.metric

    device = 'cuda:' + str(args.gpu) if cuda.is_available() else 'cpu'

    num_nodes, num_rels, num_t = get_total_number('./data/' + dataset, 'stat.txt')
    train_data, train_times = load_quadruples('./data/' + dataset, 'train.txt', num_rels)
    dev_data, dev_times = load_quadruples('./data/' + dataset, 'valid.txt', num_rels)
    test_data, test_times = load_quadruples('./data/' + dataset, 'test.txt', num_rels)

    dt_string = dataset
    main_dirName = os.path.join("TEST_RE/models/" + dt_string)
    model_path = os.path.join(main_dirName, 'models')

    is_multi = "single"
    if args.multi_step:
        is_multi = "multi"

    file_testing = open(os.path.join(main_dirName, str(k) + "_" +
                                     str(alpha) + "alpha_" + is_multi + "_testing_record_" + metric + ".txt"), "w")

    file_testing.write("Testing Start \n")
    file_testing.write("===============================\n")

    test_set = SimfyDataset(test_data)

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
            quad = data['quad'].to(device, dtype=torch.long)

            l, outputs = model(s_list, p_list, o_list)

            freq_outputs = get_outputs(
                dataset, s_list.cpu().detach().numpy().tolist(), p_list.cpu().detach().numpy().tolist(),
                t_list.cpu().detach().numpy().tolist(), num_rels, k, args.multi_step)

            freq_res = torch.softmax(freq_outputs.float(), dim=1)
            freq_res = freq_res.to(device)
            outputs = outputs * alpha + freq_res * (1 - alpha)

            if metric == 'raw':
                batch_mrr, batch_hit1, batch_hit3, batch_hit10 = calc_raw_mrr(outputs, targets, hits=[1, 3, 10])
            elif metric == 'filtered':
                batch_mrr, batch_hit1, batch_hit3, batch_hit10 = calc_filtered_test_mrr(num_nodes, outputs,
                                                                           torch.LongTensor(
                                                                               train_data),
                                                                           torch.LongTensor(
                                                                               dev_data),
                                                                           torch.LongTensor(
                                                                               test_data),
                                                                           quad,
                                                                           entity='object',
                                                                           hits=[1, 3, 10])
            elif metric == 'time_aware_filtered':
                batch_mrr, batch_hit1, batch_hit3, batch_hit10 = ta_calc_filtered_test_mrr(num_nodes, outputs,
                                                                           torch.LongTensor(
                                                                               test_data),
                                                                           quad,
                                                                           entity='object',
                                                                           hits=[1, 3, 10])


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
