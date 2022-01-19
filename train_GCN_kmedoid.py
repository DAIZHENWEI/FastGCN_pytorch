import argparse
import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import numpy as np
import pdb
import tqdm
import json
import copy
from scipy.sparse.linalg import norm as sparse_norm
import numpy.linalg as LA
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from utils import get_batches, accuracy
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data, dataloader
from models import GCN


def binary_search(cluster_size_sample, n_sampled, cluster_size):
    if len(cluster_size) == n_sampled:
        return [1] * n_sampled
    cluster_sampled = [max(1, min(int(size), upper)) for size, upper in zip(cluster_size_sample, cluster_size)]
    sum_cluster_sampled = sum(cluster_sampled)
    if sum_cluster_sampled > n_sampled:
        l, r = 0.5, 1.0
    else:
        l, r = 1.0, 1.5
    diff = 1
    while diff:
        mid = (l+r)/2
        l_sampled = [max(1, min(int(size*l), upper)) for size, upper in zip(cluster_size_sample, cluster_size)]
        new_sampled = [max(1, min(int(size*mid), upper)) for size, upper in zip(cluster_size_sample, cluster_size)]
        sum_new_sampled = sum(new_sampled)
        if sum_new_sampled > n_sampled:
            r = mid
        else:
            l = mid
        diff = sum_new_sampled - sum(l_sampled)
        print(diff)
    return l_sampled
    

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def macro_f1(pred, labels):
    labels_pred = torch.argmax(pred, dim=1).detach().cpu()
    labels = labels.detach().cpu()
    return f1_score(labels_pred, labels, average='macro')


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, nfeat, device, args)
    model.train()
    return compute_acc(pred[test_nid], labels[test_nid]), macro_f1(pred[test_nid], labels[test_nid]),  pred

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels


class GCN_Model:
    def __init__(self, args, device, data):
        self.args = args
        self.device = device
        self.train_nid, self.val_nid, self.test_nid, self.in_feats, self.labels, self.n_classes, self.feats, self.g, self.adj_norm = data
        self.model = GCN(self.in_feats, self.args.hidden, self.n_classes, self.args.dropout)
        self.model = self.model.to(device)
        self.loss_fcn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

    def run(self, epochs=None, return_embed=False):
        # Training loop
        avg = 0
        iter_tput = []
        test_acc_list = []
        test_f1_list = []
        if epochs is None:
            epochs = self.args.epochs
        
        self.feats = self.feats.to(device) 
        self.labels = self.labels.to(device)
        self.train_nid = self.train_nid.to(device)
        self.test_nid = self.test_nid.to(device)
        self.adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm).to(device)

        for epoch in range(epochs):
                
            self.model.train()
            pred = self.model(self.feats, self.adj_norm)
            loss_train = self.loss_fcn(pred[self.train_nid], self.labels[self.train_nid])
            acc_train = compute_acc(pred[self.train_nid], self.labels[self.train_nid])
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            
            self.model.eval()
            pred = self.model(self.feats, self.adj_norm)
            loss_test = self.loss_fcn(pred[self.test_nid], self.labels[self.test_nid])
            acc_test = compute_acc(pred[self.test_nid], self.labels[self.test_nid])
            f1_test = macro_f1(pred[self.test_nid], self.labels[self.test_nid])

            print(f"epchs:{epoch}~{epochs} "
                f"train_loss: {loss_train.item():.3f}, "
                f"train_acc: {acc_train.item():.3f}, "
                f"test_loss: {loss_test.item():.3f}, "
                f"test_acc: {acc_test.item():.3f}, "
                f"macro_f1_score: {f1_test:.3f}")

            test_acc_list += [acc_test.item()]
            test_f1_list += [f1_test]
        if return_embed:
            return test_acc_list, test_f1_list, pred

        return test_acc_list, test_f1_list                                                                
                                                                    


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--dataset', type=str, default='cora', help='dataset name.')
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--prop-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,5,5')
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=10)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    argparser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--seed', type=int, default=123, help='Random seed.')
    argparser.add_argument('--num-workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--ratio', type=float, default=0.5,
                        help='Proportion of samples used for training')
    argparser.add_argument('--num_cluster', type=int, default=100)
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--fix_ncluster', action='store_true', default=False, help= "fix the number of clusters")
    argparser.add_argument('--prop_to_sd', action='store_true', default=False, help= "sampled nodes from clusters proportional to cluster s.d.")
    argparser.add_argument('--remove_degree_one', action='store_true', default=False,
                        help='Recursively remove the nodes with degree one from the adjacency matrix (remove corresponding edges).')
    args = argparser.parse_args()
    

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu != -1:
        torch.cuda.manual_seed(args.seed)

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    """ Load dataset """
    loader = dataloader(args.dataset, args)
    train_index, valid_index, test_index, in_feats, labels, n_classes, feats, graph = loader.get_DGL_GCN_inputs()
    adj_norm = loader.get_norm_laplacian()


    """ Compute the feature propagation """
    laplacian = loader.get_norm_laplacian()
    laplacian = sparse_mx_to_torch_sparse_tensor(laplacian).to(device)
    feats_ppg = copy.copy(feats)
    feats_ppg = feats_ppg.to(device)
    for i in range(args.prop_layers):
        feats_ppg = torch.sparse.mm(laplacian, feats_ppg)

    feats_train = feats_ppg[train_index]
    feats_train = feats_train.detach().cpu().numpy()
    num_train = len(train_index)
    n_sampled = int(args.ratio * num_train)
    if args.fix_ncluster:
        n_cluster = min(n_sampled, args.num_cluster)
    else:
        n_cluster = n_sampled
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(feats_train)
    clusters = kmeans.labels_
    centers = kmeans.cluster_centers_
    cluster_size = [0] * n_cluster
    for i in range(n_cluster):
        cluster_size[i] = sum(clusters==i)

    cluster_size_sample = [num*args.ratio for num in cluster_size]

    cluster_sampled = binary_search(cluster_size_sample, n_sampled, cluster_size)
    print("n_sampled: {}, sum of cluster_sampled: {}".format(n_sampled, sum(cluster_sampled)))

    train_ind_sampled = []
    num_per_cluster = int(n_sampled / n_cluster)
    for i, center in enumerate(centers):
        num_per_cluster = cluster_sampled[i]
        feats_i = feats_train[clusters == i]
        kmeans_i = KMeans(n_clusters=num_per_cluster, random_state=0).fit(feats_i)
        centers_i = kmeans_i.cluster_centers_
        train_index_cluster = train_index[clusters == i]
        for center in centers_i:
            dists = LA.norm(feats_i - center, axis = 1)
            train_index_cluster_sort = train_index_cluster[dists.argsort()]
            train_ind_sampled.append(train_index_cluster_sort[0])

    train_ind_sampled = torch.from_numpy(np.sort(train_ind_sampled))
    train_ind_sampled = torch.unique(train_ind_sampled)

    labels = labels.to(device)
    feats = feats.to(device)
    data = train_ind_sampled, valid_index, test_index, in_feats, labels, n_classes, feats, graph, adj_norm



    test_accs_list, test_f1_list, num_train_sampled = [], [], []
    GraphModel = GCN_Model(args, device, data)
    test_accs, test_f1 = GraphModel.run()


    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)



    np.save(directory + 'GCN_accuracy_list_{}_Kmedoid_ncluster{}_PropToSd{}_AggregateLayer{}_ratio{}.npy'.format(args.dataset, 
                                                                                                            n_cluster,
                                                                                                            args.prop_to_sd,
                                                                                                            args.prop_layers, 
                                                                                                            args.ratio), test_accs)

    np.save(directory + 'GCN_micro_f1_list_{}_Kmedoid_ncluster{}_PropToSd{}_AggregateLayer{}_ratio{}.npy'.format(args.dataset, 
                                                                                                            n_cluster,
                                                                                                            args.prop_to_sd,
                                                                                                            args.prop_layers, 
                                                                                                            args.ratio), test_f1)
