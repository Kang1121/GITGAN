import argparse
import h5py
import numpy as np
from os.path import join as pjoin
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn.init as init
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


CHANNELS_62 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
               'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10',
               'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h',
               'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8',
               'PO3', 'PO4']
CHANNELS_39 = ['F3', 'Fz', 'F4', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
               'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
               'P1', 'P2', 'POz', 'TP7', 'TP8', 'PO3', 'PO4']
RESERVED = sorted([CHANNELS_62.index(ch) for ch in CHANNELS_39])


def read_epsilon(dataset, proportion):
    with open('./dbscan/{}{}.txt'.format(dataset, proportion), 'r') as f:
        epsilon_values = [
            [float(val) for val in line.strip().strip(',').split(',')]
            for line in f.readlines()
        ]
    return epsilon_values


def process_cluster(x, y, epsilon, sub_idx):
    num = len(x) // (max(y) + 1)
    results = [
        process_sub_cluster(x[num * i: num * (i + 1)], i, epsilon, sub_idx)
        for i in range(max(y) + 1)
    ]
    list_x, list_y = zip(*results)
    return list(list_x), list(list_y)


def process_sub_cluster(x_, cluster_idx, epsilon, sub_idx):
    p = epsilon[sub_idx[0] - 1][cluster_idx]
    temp = StandardScaler().fit_transform(np.mean(x_, axis=1))
    pca = PCA(n_components=10)
    temp = pca.fit_transform(temp)
    lbs = DBSCAN(eps=p, min_samples=5).fit(temp).labels_
    return x_[lbs != -1], np.ones((len(lbs[lbs != -1])), dtype=np.int64) * cluster_idx


def cbaDBSCAN(args, data, label, sub_idx):
    epsilon = read_epsilon(args.dataset, args.proportion)
    list_x, list_y = [], []
    for x, y in zip(data, label):
        sorted_order = np.argsort(y)
        x, y = x[sorted_order], y[sorted_order]
        cluster_x, cluster_y = process_cluster(x, y, epsilon, sub_idx)
        list_x.extend(cluster_x)
        list_y.extend(cluster_y)
    return list_x, list_y


def get_data(dfile, subj):
    xs = []
    ys = []
    for i in subj:
        dpath = '/s' + str(i)
        x, y = dfile[pjoin(dpath, 'X')], dfile[pjoin(dpath, 'Y')]
        x = x[:, RESERVED, :]
        xs.append(x[:]), ys.append(y[:])

    return xs, ys


def dataloader(args, train_subjs, valid_index, test__index, cv_set):
    data = h5py.File(args.data_path, 'r')
    test_subj = [i for i in range(1, args.num_subjects + 1) if i not in train_subjs]
    x_train, y_train = get_data(data, train_subjs)
    if args.outrm:
        x_train, y_train = cbaDBSCAN(args, x_train, y_train, test_subj)
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

    x_test, y_test = get_data(data, test_subj)
    x_valid, y_valid = x_test[0][cv_set[valid_index]], y_test[0][cv_set[valid_index]]
    x__test, y__test = x_test[0][cv_set[test__index]], y_test[0][cv_set[test__index]]

    Xs = list(zip(torch.from_numpy(x_train.transpose((0, 2, 1))).unsqueeze(1).float(), torch.from_numpy(y_train).long()))
    Xt = list(zip(torch.from_numpy(x_valid.transpose((0, 2, 1))).unsqueeze(1).float(), torch.from_numpy(y_valid).long()))

    train_set = PairedDataset(Xs, Xt)
    test__set = TensorDataset(torch.from_numpy(x__test.transpose((0, 2, 1))).unsqueeze(1).float(), torch.from_numpy(y__test).long())
    train_sampler = DistributedSampler(train_set) if args.use_ddp else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler) if args.use_ddp else DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test__loader = DataLoader(test__set, batch_size=1, shuffle=False)

    return train_loader, train_sampler, test__loader


class PairedDataset(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1  # data1 is Xs
        self.data2 = data2  # data2 is Xt

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        x1, y1 = self.data1[idx]
        # radomly select a sample from dataset2
        idx = np.random.randint(0, len(self.data2))
        x2, y2 = self.data2[idx]
        return (x1, y1, x2, y2)


def create_argparser(config_file=None):
    defaults = dict()
    if config_file is not None:
        defaults = config_file
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def kaiming_normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
