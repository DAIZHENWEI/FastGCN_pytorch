import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse.linalg import norm as sparse_norm
import numpy as np
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import pdb

from models import GCN
from sampler import Sampler_FastGCN, Sampler_ASGCN, Sampler_LADIES, Sampler_Random
from utils import load_data, accuracy, HLoss, get_batches
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
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--pre_train_epochs', type=int, default=200,
                        help='Number of pre-training epochs.')
    parser.add_argument('--comp_embed_epochs', type=int, default=800,
                        help='Number of epochs to compute embeddings.')
    parser.add_argument('--epochs', type=int, default=800,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--pretrain_ratio', type=float, default=0.01,
                        help='Proportion of samples used for training')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='Proportion of samples used for training')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args



def get_batches_active(train_ind, train_labels, probs, prob_thres, batch_size=128, shuffle=True):
    """
    Inputs:
        train_ind: np.array
    """
    train_ind = train_ind[probs>=prob_thres]
    nums = train_ind.shape[0]
    if shuffle:
        np.random.shuffle(train_ind)
    i = 0
    while i < nums:
        cur_ind = train_ind[i:i + batch_size]
        cur_labels = train_labels[cur_ind]
        yield cur_ind, cur_labels
        i += batch_size



def pre_train(train_ind, train_labels, batch_size, train_times, train_probs, pretrain_ratio):
    """
    Inputs:
        pretrain_ratio: the proportion of the training samples selected
    """
    t = time.time()
    model.train()
    prob_thres = np.quantile(train_probs, 1-pretrain_ratio)
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches_active(train_ind, train_labels, train_probs, prob_thres, batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(batch_inds)
            optimizer.zero_grad()
            output = model(sampled_feats, sampled_adjs)
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def train(train_ind, train_labels, batch_size, train_times):
    """
    Inputs:
        pretrain_ratio: the proportion of the training samples selected
    """
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind, train_labels, batch_size):
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
    outputs = model(test_feats, test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


if __name__ == '__main__':
    # load data, set superpara and constant
    args = get_args()
    """
    adj: Laplacian matrix of the whole graph
    adj_origin: Adjacency matrix of the whole graph
    """
    adj, features, adj_train, train_features, y_train, y_test, test_index, adj_origin = load_data(args.dataset)
    
    # layer_sizes = [128, 128]
    layer_sizes = [args.batchsize, args.batchsize]
    input_dim = features.shape[1]
    num_nodes = adj.shape[0]
    train_nums = adj_train.shape[0]
    test_gap = args.test_gap
    nclass = y_train.shape[1]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # set device
    if args.cuda:
        device = torch.device("cuda")
        print("use cuda")
    else:
        device = torch.device("cpu")

    # data for train and test
    train_index = np.setdiff1d(np.arange(num_nodes), test_index)
    features = torch.FloatTensor(features).to(device)
    train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]

    col_norm = sparse_norm(adj_train, axis=0)
    train_probs = col_norm / np.sum(col_norm)

    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device) for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    # pdb.set_trace()

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
    train_samples_list = []
    prob_thres = np.quantile(train_probs, 1-args.pretrain_ratio)
    train_ind_pretrain = train_index[train_probs>=prob_thres]
    for epochs in range(0, args.pre_train_epochs // test_gap):
        train_loss, train_acc, train_time = train(train_ind_pretrain,
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap)
        test_loss, test_acc, test_time = test(test_adj,
                                              test_feats,
                                              test_labels,
                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_samples: {len(train_ind_pretrain)} "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s")
        test_acc_list += [test_acc]


    train_adj = [adj, adj[train_index, :]]
    train_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device) for cur_adj in train_adj]
    Entropy = HLoss()

    """ Continue training """
    train_ind_cur = train_ind_pretrain
    for epochs in range(0, (args.epochs - args.pre_train_epochs) // test_gap):
        if epochs % 20 == 0:
            """ Compute Embedding Vectors """
            model.eval()
            embeds = model(features, train_adj, compute_embed=True)
            embeds = embeds.detach().cpu().numpy()
            kmedoids = KMedoids(n_clusters=int(num_nodes*args.ratio), random_state=0).fit(embeds)
            center_sum = np.sum(kmedoids.cluster_centers_, axis = 1)
            embeds_sum = np.sum(embeds, axis = 1)
            extra_train_index = np.array([i for i, x in enumerate(embeds_sum) if x in center_sum])
            extra_train_index = train_index[extra_train_index]
        train_ind_cur = np.unique(np.concatenate((train_ind_cur, extra_train_index))) 
        train_loss, train_acc, train_time = train(train_ind_cur,
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap)
        test_loss, test_acc, test_time = test(test_adj,
                                              test_feats,
                                              test_labels,
                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_samples: {len(train_ind_cur)} "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s")
        test_acc_list += [test_acc]
        train_samples_list += [len(train_ind_cur)]

    np.save('./save/test_accuracy_list_{}_{}_active_laplacian_embed_cluster_increment_compround{}_pretrain{}_ratio{}.npy'.format(args.dataset, 
                                                                                                    args.model,
                                                                                                    args.comp_embed_epochs,
                                                                                                    args.pre_train_epochs, 
                                                                                                    args.pretrain_ratio), test_acc_list)

    np.save('./save/sample_count_list_{}_{}_active_laplacian_embed_cluster_increment_compround{}_pretrain{}_ratio{}.npy'.format(args.dataset, 
                                                                                                    args.model,
                                                                                                    args.comp_embed_epochs,
                                                                                                    args.pre_train_epochs, 
                                                                                                    args.pretrain_ratio), train_samples_list)