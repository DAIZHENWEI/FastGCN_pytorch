import argparse
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse.linalg import norm as sparse_norm
import numpy as np
import pdb

from models import GCN, GCN4, GCN3
from sampler import Sampler_FastGCN, Sampler_ASGCN, Sampler_LADIES, Sampler_GraphSage, Sampler_Random
from utils import load_data, accuracy
from utils import sparse_mx_to_torch_sparse_tensor


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Fast',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=1,
                        help='the train epochs between two test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use. -1 means')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Proportion of samples used for training')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    parser.add_argument('--remove_degree_one', action='store_true', default=False,
                        help='Recursively remove the nodes with degree one from the adjacency matrix (remove corresponding edges).')
    parser.add_argument('--exclude_high_degree', action='store_true', default=False,
                        help='Do not consider the extremely large degree nodes')
    args = parser.parse_args()
    return args


def get_batches_active(train_labels, train_ind_sort, ratio, batch_size=128, shuffle=True):
    """
    Inputs:
        train_ind: np.array
        ratio: proportion of training samples picked
    """
    train_ind = train_ind_sort[:int(ratio*len(train_ind_sort))]
    train_ind = np.sort(train_ind)
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size



def train(train_ind, train_labels, batch_size, train_times, train_ind_sort, ratio):
    """
    Inputs:
        ratio: the proportion of the training samples selected
    """
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches_active(train_labels, train_ind_sort, ratio, batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(batch_inds)
            optimizer.zero_grad()
            output = model(sampled_feats, sampled_adjs)
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def test(test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj, test=True)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


if __name__ == '__main__':
    # load data, set superpara and constant
    args = get_args()
    adj, features, adj_train, train_features, y_train, y_test, test_index, adj_origin = load_data(args.dataset, args)
    num_train = len(y_train)
    col_norm = sparse_norm(adj_train, axis=0)
    if args.exclude_high_degree:
        col_degree = sparse_norm(adj_origin, axis=0)
        train_ind_degree_sort = np.arange(num_train)[col_degree.argsort()[::-1]]
        high_degree_nodes = train_ind_degree_sort[:int(0.01*num_train)]
        col_norm[high_degree_nodes] = 0
    train_probs = col_norm / np.sum(col_norm)

    """ need to get the original adjacency matrix for model training"""
    if args.remove_degree_one:
        args.remove_degree_one = False
        adj, features, adj_train, train_features, y_train, y_test, test_index, adj_origin = load_data(args.dataset, args)
        args.remove_degree_one = True

    # layer_sizes = [128, 128]
    layer_sizes = [args.batchsize, args.batchsize]
    if args.dataset == 'reddit':
        layer_sizes = [args.batchsize] * 4
    if args.dataset == 'ogbn_arxiv':
        layer_sizes = [args.batchsize] * 3
    input_dim = features.shape[1]
    train_nums = adj_train.shape[0]
    test_gap = args.test_gap
    nclass = y_train.shape[1]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu != -1:
        torch.cuda.manual_seed(args.seed)
    # set device
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # data for train and test
    features = torch.FloatTensor(features).to(device)
    train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]



    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device) for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    # init the sampler
    if args.model == 'Fast':
        sampler = Sampler_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)
    elif args.model == 'LADIES':
        sampler = Sampler_LADIES(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)  
    elif args.model == 'GraphSage':
        sampler = Sampler_GraphSage(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device)       
    elif args.model == 'AS':
        sampler = Sampler_ASGCN(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                device=device)
    elif args.model == 'Random':
        sampler = Sampler_Random(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                device=device)
    else:
        print(f"model name error, no model named {args.model}")
        exit()

    # init model, optimizer and loss function
    if args.dataset == 'reddit':
        model = GCN4(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=args.dropout,
                    sampler=sampler).to(device)
    elif args.dataset == 'ogbn_arxiv':
        model = GCN3(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=args.dropout,
                    sampler=sampler).to(device)
    else:
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=nclass,
                    dropout=args.dropout,
                    sampler=sampler).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss

    # train and test
    test_acc_list = []
    train_ind_sort = np.arange(train_nums)[train_probs.argsort()[::-1]]
    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(np.arange(train_nums),
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap, 
                                                  train_ind_sort,
                                                  args.ratio)
        test_loss, test_acc, test_time = test(test_adj,
                                              test_feats,
                                              test_labels,
                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s")
        test_acc_list += [test_acc]

    if args.remove_degree_one:
        np.save('./save/test_accuracy_list_{}_{}_active_laplacian_remove_degree_one_ratio{}.npy'.format(args.dataset, args.model, args.ratio), test_acc_list)
    elif args.exclude_high_degree:
        np.save('./save/test_accuracy_list_{}_{}_active_laplacian_exclude_high_degree_ratio{}.npy'.format(args.dataset, args.model, args.ratio), test_acc_list)
    else:
        np.save('./save/test_accuracy_list_{}_{}_active_laplacian_ratio{}.npy'.format(args.dataset, args.model, args.ratio), test_acc_list)