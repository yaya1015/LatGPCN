import sys
import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import scipy.io as sio
from torch_geometric.utils import remove_self_loops, add_self_loops


class Data(object):
    def __init__(self, dataset_name, adj, features, labels, idx_train, idx_val, idx_test):
        self.dataset_name = dataset_name
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test

def load_wiki():
    def normalize_adj(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
        return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

    data = sio.loadmat("./data/wiki.mat")
    adj = data['W']
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj-sp.csr_matrix(np.diag(np.diag(adj.todense())))+sp.eye(adj.shape[0]))
    adj = (adj + adj.T)/2.

    features = data['fea']
    labels = data['gnd'].flatten()-1
    train_idx, val_idx, test_idx = split_data(torch.tensor(labels).long(),20,400)
    data = Data("wiki", adj, features, labels, train_idx, val_idx, test_idx)
    return data


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask

def split_data(labels, n_train_per_class, n_val):
    n_class = int(torch.max(labels)) + 1
    train_idx = np.array([], dtype=np.int64)
    remains = np.array([], dtype=np.int64)
    for c in range(n_class):
        candidate = torch.nonzero(labels == c).T.numpy()[0]
        np.random.shuffle(candidate)
        train_idx = np.concatenate([train_idx, candidate[:n_train_per_class]])
        remains = np.concatenate([remains, candidate[n_train_per_class:]])
    np.random.shuffle(remains)
    val_idx = remains[:n_val]
    test_idx = remains[n_val:]
    train_mask = index_to_mask(train_idx, labels.size(0))
    val_mask = index_to_mask(val_idx, labels.size(0))
    test_mask = index_to_mask(test_idx, labels.size(0))
    return train_mask, val_mask, test_mask

def preprocess(adj, features, labels, device='cpu', features_norm=True, adj_norm=False):
    labels = torch.LongTensor(labels)
    
    features = normalize_feature(features) if features_norm else features
    features = torch.FloatTensor(np.array(features.todense()))

    adj = adj.tocoo().astype(np.float32)
    edge = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    value = torch.from_numpy(adj.data)
    edge, value = remove_self_loops(edge, value)
    edge, value = add_self_loops(edge, value,num_nodes=features.size(0))

    return edge.to(device), value.to(device), features.to(device), labels.to(device)

def normalize_feature(mx):
    """Row-normalize sparse matrix"""
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx