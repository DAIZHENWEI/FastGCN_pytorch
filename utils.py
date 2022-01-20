import sys
import pdb
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.data
import numpy as np
import networkx as nx
import scipy
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from ogb.nodeproppred import DglNodePropPredDataset

def _load_data(dataset_str):
    """Load data."""

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally) )
    idx_val = range(len(ally)-0, len(ally))
    # idx_train = range(len(ally)-500)
    # idx_val = range(len(ally)-500, len(ally))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val   = np.zeros(labels.shape)
    y_test  = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :]     = labels[val_mask, :]
    y_test[test_mask, :]   = labels[test_mask, :]

    return (adj, features, y_train, y_val, y_test,
            train_mask, val_mask, test_mask)


def nontuple_preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    ep = 1e-10
    r_inv = np.power(rowsum + ep, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def nontuple_preprocess_adj(adj):
    adj_normalized = normalize_adj(sp.eye(adj.shape[0]) + adj)
    # adj_normalized = sp.eye(adj.shape[0]) + normalize_adj(adj)
    return adj_normalized.tocsr()


def load_data(dataset, args):
    if dataset == "reddit":
        return load_reddit(args)
    if dataset == "ogbn_arxiv":
        return load_ogbn_arxiv(args)
    # train_mask, val_mask, test_mask: np.ndarray, [True/False] * node_number
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        _load_data(dataset)
    # pdb.set_trace()
    train_index = np.where(train_mask)[0]
    adj_train = adj[train_index, :][:, train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]

    num_train = adj_train.shape[0]

    features = nontuple_preprocess_features(features).todense()
    train_features = features[train_index]

    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)

    if dataset == 'pubmed':
        norm_adj = 1*sp.diags(np.ones(norm_adj.shape[0])) + norm_adj
        norm_adj_train = 1*sp.diags(np.ones(num_train)) + norm_adj_train

    return (norm_adj, features, norm_adj_train, train_features,
            y_train, y_test, test_index, adj_train)



def load_reddit(args):
    dataset = dgl.data.RedditDataset()
    g = dataset[0]
    node = g.ndata
    edge = g.edges()
    num_node = len(node['train_mask'])
    num_train = torch.sum(node['train_mask'])
    feats =  node['feat'].detach().numpy()
    labels = node['label']
    num_labels = len(torch.unique(num_labels))
    labels = F.one_hot(labels, num_classes=num_labels).detach().numpy()
    """ Get adjacency matrix from edge node pairs"""
    row = edge[1].detach().numpy()
    col = edge[0].detach().numpy()
    dat = np.ones((len(row)))
    print("========= Generating adjacency matrix ===========")
    adj = csr_matrix((dat, (row, col)), shape=(num_node, num_node))
    """ Get training and testing samples index """
    train_index = np.arange(num_node)[node['train_mask']]
    test_index = np.arange(num_node)[node['test_mask']]

    adj_train = adj[train_index, :][:, train_index]
    y_train = labels[train_index]
    y_test = labels[test_index]
    train_features = feats[train_index]
    
    """ Normalize the adjacency matrix """
    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)   
    norm_adj = 1*sp.diags(np.ones(norm_adj.shape[0])) + norm_adj
    norm_adj_train = 1*sp.diags(np.ones(num_train)) + norm_adj_train
    return (norm_adj, feats, norm_adj_train, train_features, y_train, 
            y_test, test_index, adj_train)


def load_ogbn_arxiv(args):
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    g, node_labels = dataset[0]
    g = dgl.add_reverse_edges(g)
    g.ndata['label'] = node_labels[:, 0]
    node = g.ndata
    edge = g.edges()    

    feats = g.ndata['feat']
    idx_split = dataset.get_idx_split()
    train_index = idx_split['train']
    valid_index = idx_split['valid']
    test_index = idx_split['test']

    num_node = feats.shape[0]
    num_train = len(train_index)
    feats =  node['feat'].detach().numpy()
    labels = node['label']
    labels = F.one_hot(labels, num_classes=40).detach().numpy()
    """ Get adjacency matrix from edge node pairs"""
    row = edge[1].detach().numpy()
    col = edge[0].detach().numpy()
    dat = np.ones((len(row)))
    print("========= Generating adjacency matrix ===========")
    adj = csr_matrix((dat, (row, col)), shape=(num_node, num_node))

    if args.remove_degree_one:
        adj, sample_ind, row_s, col_s = remove_degree_one(adj, row, col)
        dat = np.ones((len(row_s)))
        print("========= Generating pruned adjacency matrix ===========")
        adj = csr_matrix((dat, (row_s, col_s)), shape=(num_node, num_node)) 
               
    adj_train = adj[train_index, :][:, train_index]
    y_train = labels[train_index]
    y_test = labels[test_index]
    train_features = feats[train_index]

    """ Normalize the adjacency matrix """
    norm_adj_train = nontuple_preprocess_adj(adj_train)
    norm_adj = nontuple_preprocess_adj(adj)   
    norm_adj = 1*sp.diags(np.ones(norm_adj.shape[0])) + norm_adj
    norm_adj_train = 1*sp.diags(np.ones(num_train)) + norm_adj_train
    return (norm_adj, feats, norm_adj_train, train_features, y_train, 
            y_test, test_index, adj_train)



def remove_degree_one(adj, row, col):
    num_remove = 1
    ind = np.arange(adj.shape[0])
    while num_remove > 0:
        dim1 = adj.shape[0]
        col_sum = scipy.sparse.csr_matrix.sum(adj, axis = 0)
        col_sum = np.array(col_sum)[0]
        non_one_ix = np.nonzero(col_sum - 1)[0]
        one_ind = np.delete(ind, non_one_ix)
        ind = ind[non_one_ix]
        adj = adj[non_one_ix,: ]
        adj = adj[:, non_one_ix]
        num_remove = dim1 - adj.shape[0]
        
        index_remove = np.intersect1d(row, one_ind, return_indices = True)
        print(len(index_remove[1]), len(one_ind))
        row = np.delete(row, index_remove[1])
        col = np.delete(col, index_remove[1])
        
        index_remove = np.intersect1d(col, one_ind, return_indices = True)
        row = np.delete(row, index_remove[1])
        col = np.delete(col, index_remove[1])
    return adj, ind, row, col



def get_batches(train_ind, train_labels, batch_size=64, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = torch.exp(x) * x
        b = -1.0 * b.sum(dim=1)
        return b


class Entropy_loss(nn.Module):
    def __init__(self):
        super(Entropy_loss, self).__init__()

    def forward(self, x):
        probs = F.softmax(x, dim=1)
        b = torch.log(probs) * probs
        b = -1.0 * b.sum(dim=1)
        return b



if __name__ == '__main__':
    pdb.set_trace()
    adj, features, adj_train, train_features, y_train, y_test, test_index = \
        load_data('cora')
    pdb.set_trace()


