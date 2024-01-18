import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SimfyDataset(Dataset):
    """
    prepare dataset
    """

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

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self, ):
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / math.sqrt(self.dim) ** np.linspace(0, (self.dim - 1) / math.sqrt(self.dim), self.dim,
                                                                     dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output

class TimeSimfy(nn.Module):
    def __init__(self, num_e, num_rel, num_t, embedding_dim):
        super(TimeSimfy, self).__init__()

        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel

        # embedding initiation
        self.rel_embeds = nn.Parameter(torch.zeros(2 * self.num_rel, embedding_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.similarity_pred_layer = nn.Linear(3 * embedding_dim, embedding_dim)
        self.weights_init(self.similarity_pred_layer)

        # time integrated
        self.time_encode = TimeEncode(embedding_dim)
        self.time_encode.reset_parameters()

        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, s, p, o, t):
        t_embeds = self.time_encode(t.float())
        preds_raw = self.tanh(self.similarity_pred_layer(self.dropout(torch.cat((self.entity_embeds[s],
                                                                                 self.rel_embeds[p], t_embeds), dim=1))))
        preds = F.softmax(preds_raw.mm(self.entity_embeds.transpose(0, 1)), dim=1)

        nce_loss = torch.sum(torch.gather(torch.log(preds), 1, o.view(-1, 1)))
        nce_loss /= -1. * o.shape[0]

        return nce_loss, preds

class Simfy(nn.Module):
    def __init__(self, num_e, num_rel, num_t, embedding_dim):
        super(Simfy, self).__init__()

        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel

        # embedding initiation
        self.rel_embeds = nn.Parameter(torch.zeros(2 * self.num_rel, embedding_dim))
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, embedding_dim))
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.similarity_pred_layer = nn.Linear(2 * embedding_dim, embedding_dim)
        self.weights_init(self.similarity_pred_layer)

        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()


    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, s, p, o):
        preds_raw = self.tanh(self.similarity_pred_layer(self.dropout(torch.cat((self.entity_embeds[s],
                                                                                 self.rel_embeds[p]), dim=1))))
        preds = F.softmax(preds_raw.mm(self.entity_embeds.transpose(0, 1)), dim=1)

        nce_loss = torch.sum(torch.gather(torch.log(preds), 1, o.view(-1, 1)))
        nce_loss /= -1. * o.shape[0]

        return nce_loss, preds


