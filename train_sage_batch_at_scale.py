import argparse
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import dgl.nn.pytorch as dglnn
import numpy as np
import numpy.linalg as LA
import collections
import random
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import copy
import pdb
import tqdm
from scipy.sparse.linalg import norm as sparse_norm
from utils import get_batches, accuracy, Entropy_loss
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data, margin_score, dataloader
from models import SAGE

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
    
def random_round_robin(Kt, candidates, cluster_dict):
    if len(candidates) == Kt:
        return np.array(candidates)
    cand_clusters=collections.defaultdict(list)
    # pdb.set_trace()
    for cand in candidates:
        cand_clusters[cluster_dict[cand]].append(cand)

    for key in cand_clusters:
        random.shuffle(cand_clusters[key])
    S = []
    i = 0
    while i < Kt:
        for key in cand_clusters:
            if cand_clusters[key]:
                item = cand_clusters[key].pop()
                S.append(item)
                i += 1
    return np.array(S)


def Kmeans_sample(Kt, candidates, feats):
    if len(candidates) == Kt:
        return np.array(candidates)
    feats_sample = feats[candidates]
    kmeans = KMeans(n_clusters=Kt, random_state=0).fit(feats_sample)
    Kcenters = kmeans.cluster_centers_
    train_ind_sampled = []
    for center in Kcenters:
        dists = LA.norm(feats_sample - center, axis = 1)
        train_index_sort = candidates[dists.argsort()]
        train_ind_sampled.append(train_index_sort[0])
    return np.array(train_ind_sampled)



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

class GraphSage:
    def __init__(self, args, device, data):
        self.args = args
        self.device = device
        self.train_nid, self.val_nid, self.test_nid, self.in_feats, self.labels, self.n_classes, self.nfeat, self.g = data
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(
                                    [int(fanout) for fanout in args.fan_out.split(',')])
        self.model = SAGE(self.in_feats, self.args.hidden, self.n_classes, self.args.num_layers, F.relu, self.args.dropout)
        self.model = self.model.to(device)
        self.loss_fcn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd)

    def run(self, epochs=None, return_embed=False):
        # Training loop
        avg = 0
        iter_tput = []
        test_acc_list = []
        test_f1_list = []
        dataloader = self.dataloader(self.args.batchsize, shuffle=True)
        if epochs is None:
            epochs = self.args.epochs
        for epoch in range(epochs):
            tic = time.time()
            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                tic_step = time.time()
                # copy block to gpu
                blocks = [blk.int().to(self.device) for blk in blocks]
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(self.nfeat, self.labels, seeds, input_nodes)

                # Compute loss and prediction
                batch_pred = self.model(blocks, batch_inputs)
                loss = self.loss_fcn(batch_pred, batch_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            toc = time.time()
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch % args.eval_every == 0:
                test_acc, test_f1, pred = self.evaluation()
                test_acc_list += [test_acc.cpu().numpy()]
                test_f1_list += [test_f1]
        if return_embed:
            return test_acc_list, test_f1_list, pred

        return test_acc_list, test_f1_list 

    def compute_gradient(self):
        dataloader = self.dataloader(batchsize=1, shuffle=False)
        grad_norm = []
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()
            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(self.nfeat, self.labels, seeds, input_nodes)

            # Compute loss and prediction
            batch_pred = self.model(blocks, batch_inputs)
            loss = self.loss_fcn(batch_pred, batch_labels)
            self.optimizer.zero_grad()
            loss.backward()
            norm = sum([torch.norm(p.grad).item() for p in self.model.parameters()])
            grad_norm.append(norm)
            if step % 200 == 0:
                print('Number of steps {:.4f}'.format(step))
        return np.array(grad_norm)


    def evaluation(self):
        test_acc, test_f1, pred = evaluate(self.model, self.g, self.nfeat, self.labels, self.val_nid, self.test_nid, self.device)
        print('Test Acc {:.4f}'.format(test_acc))
        print('Test Macro F1 {:.4f}'.format(test_f1))
        return test_acc, test_f1, pred

    def dataloader(self, batchsize, shuffle = True):
        return dgl.dataloading.NodeDataLoader(self.g, self.train_nid, self.sampler, batch_size=batchsize,
                                                        shuffle=shuffle, drop_last=False, num_workers=self.args.num_workers)                                                                    
                                                                    



if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--dataset', type=str, default='cora', help='dataset name.')
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--pre_train_epochs', type=int, default=200, help='Number of pre-training epochs.')
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
    argparser.add_argument('--test_batchsize', type=int, default=2048, help='batchsize for testing')
    argparser.add_argument('--sample_freq', type=int, default=40, help = 'frequnecy of resampling nodes with large uncertainties')
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--seed', type=int, default=123, help='Random seed.')
    argparser.add_argument('--num-workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--pretrain_ratio', type=float, default=0.01, help='Proportion of samples used for training')
    argparser.add_argument('--ratio', type=float, default=0.01, help='Proportion of samples used for training')
    argparser.add_argument('--total_ratio', type=float, default=0.05, help='Proportion of total number of samples used for training')
    argparser.add_argument('--num_cluster', type=int, default=200)
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=1e-4)
    argparser.add_argument('--fix_ncluster', action='store_true', default=False, help= "fix the number of clusters")
    argparser.add_argument('--prop_to_sd', action='store_true', default=False, help= "sampled nodes from clusters proportional to cluster s.d.")
    argparser.add_argument('--sample_method', type=str, default='kmeans', help = "method to pick nodes from candidate set")
    argparser.add_argument('--cold_start', action='store_true', default=False, help= "do not use the warm start")
    argparser.add_argument('--redundancy', type=int, default=20, help = "redundancy = Km/Kt")
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

    prop_to_sd, prop_to_cluster_size = False, False
    if args.prop_to_sd:
        prop_to_sd = True
    else:
        prop_to_cluster_size = True

    """ Load dataset """
    loader = dataloader(args.dataset, args)
    train_index, valid_index, test_index, in_feats, labels, n_classes, feats, graph = loader.get_DGL_GCN_inputs()

    """ Compute the feature propagation """
    laplacian = loader.get_norm_laplacian()
    laplacian = sparse_mx_to_torch_sparse_tensor(laplacian)
    feats_ppg, feats_input = copy.copy(feats), copy.copy(feats)
    for i in range(args.prop_layers):
        feats_ppg = torch.sparse.mm(laplacian, feats_ppg)
    del laplacian
    torch.cuda.empty_cache()
    feats_ppg = feats_ppg.to(device)

    feats_train = feats_ppg[train_index]
    feats_test = feats_ppg[test_index]
    feats_input_train = feats_input[train_index]
    feats_ppg = feats_ppg.detach().cpu().numpy()
    feats_input = feats_input.detach().cpu().numpy()
    feats_train = feats_train.detach().cpu().numpy()
    feats_test = feats_test.detach().cpu().numpy()
    num_train = len(train_index)
    num_test = len(test_index)
    train_index = train_index.numpy()
    test_index = test_index.numpy()
    n_sampled = int(args.pretrain_ratio * num_train)
    
    # train_ind_sampled = np.random.choice(train_index, n_sampled, replace=False)
    # train_ind_sampled = torch.from_numpy(train_ind_sampled)
    # print("number of sampled nodes: {}:".format(n_sampled))

    if args.fix_ncluster:
        n_cluster = min(n_sampled, args.num_cluster)
    else:
        n_cluster = n_sampled

    if args.sample_method == "cluster_input":
        kmeans_train = KMeans(n_clusters=n_cluster, random_state=0).fit(feats_input_train)
    else:
        kmeans_train = KMeans(n_clusters=n_cluster, random_state=0).fit(feats_train)
    
    clusters = kmeans_train.labels_
    centers = kmeans_train.cluster_centers_
    train_index_cluster_dict = collections.defaultdict(int)
    for i in range(num_train):
        train_index_cluster_dict[train_index[i]] = clusters[i]
    cluster_size = [0] * n_cluster
    for i in range(n_cluster):
        cluster_size[i] = sum(clusters==i)

    # ### Use the clustering of test set to predict the training node labels
    # clusters_test = kmeans_test.predict(feats_train)
    # test_index_cluster_dict = collections.defaultdict(int)
    # for i in range(num_train):
    #     test_index_cluster_dict[train_index[i]] = clusters_test[i]

    if prop_to_sd:
        dists = []
        for i in range(n_cluster):
            feats_i = feats_train[clusters == i]
            dists.append(np.sum(LA.norm(feats_i - centers[i], axis = 1, ord=1)))
        cluster_size_sample = [dist*num_train*args.pretrain_ratio/np.sum(dists) for dist in dists]
    else:
        cluster_size_sample = [num*args.pretrain_ratio for num in cluster_size]

    cluster_sampled = binary_search(cluster_size_sample, n_sampled, cluster_size)
    print("n_sampled: {}, sum of cluster_sampled: {}".format(n_sampled, sum(cluster_sampled)))

    train_ind_sampled = []
    num_per_cluster = int(n_sampled / n_cluster)
    for i, center in enumerate(centers):
        num_per_cluster = cluster_sampled[i]
        if args.sample_method == "cluster_input":
            feats_i = feats_input_train[clusters == i]
        else:
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
    data = train_ind_sampled, valid_index, test_index, in_feats, labels, n_classes, feats, graph

    # pdb.set_trace()

    """ Pre-training model """
    test_accs_list, test_f1_list, num_train_sampled = [], [], []
    GraphModel = GraphSage(args, device, data)
    test_accs, test_f1, pred = GraphModel.run(epochs=args.pre_train_epochs, return_embed=True)
    test_accs_list += test_accs
    test_f1_list += test_f1
    num_train_sampled = [n_sampled] * args.pre_train_epochs

    """ Continue training """
    Entropy = Entropy_loss()
    sample_freq = args.sample_freq
    num_total_sampled = int(num_train * args.total_ratio)
    epochs = args.pre_train_epochs
    redundancy = args.redundancy
    while len(train_ind_sampled) < num_total_sampled: 
        entropy_score = Entropy(pred).detach().cpu().numpy()
        # entropy_score = margin_score(pred).detach().cpu().numpy()
        Kt = int(len(train_index) * args.ratio)
        if num_total_sampled - len(train_ind_sampled) < Kt:
            Kt = num_total_sampled - len(train_ind_sampled)
        Km = Kt * redundancy
        train_entropy = entropy_score[train_index]
        train_index_sort = torch.from_numpy(train_index[train_entropy.argsort()[::-1]])
        candidates = train_index_sort[:Km]
        if args.sample_method == "round_robin":
            train_ind_new = random_round_robin(Kt, candidates, train_index_cluster_dict)
        elif args.sample_method == "random_choice":
            train_ind_new = np.random.choice(candidates.numpy(), Kt, replace=False)
        elif args.sample_method == "cluster_input":
            train_ind_new = Kmeans_sample(Kt, candidates, feats_input)
        elif args.sample_method == "cluster_embed":
            pred = pred.detach().cpu().numpy()
            train_ind_new = Kmeans_sample(Kt, candidates, pred)
        else:
            train_ind_new = Kmeans_sample(Kt, candidates, feats_ppg)
        
        train_ind_new_sampled = np.intersect1d(train_ind_new, train_ind_sampled)
        train_ind_sampled = np.unique(np.concatenate((train_ind_sampled, train_ind_new)))
        print("Number of newly picked samples: {}, Number of total selected samples: {}".format(len(train_ind_new), len(train_ind_sampled)))
        print("Number of large entropy nodes already sampled: {}".format(len(train_ind_new_sampled)))
        # if args.cold_start: 
        #     GraphModel = GraphSage(args, device, data)
        GraphModel.train_nid = train_ind_sampled
        test_accs, test_f1, pred = GraphModel.run(epochs=sample_freq, return_embed=True)
        test_accs_list += test_accs
        test_f1_list += test_f1
        num_train_sampled += [len(train_ind_sampled)] * sample_freq   
        epochs += sample_freq 
    
    if not args.cold_start:
        test_accs, test_f1 = GraphModel.run(epochs=200)
        test_accs_list += test_accs
        test_f1_list += test_f1
        # num_train_sampled += [len(train_ind_sampled)] * (args.epochs-epochs)     
    else:
        ### Do not use warm start, training from stratch
        GraphModel = GraphSage(args, device, data)
        GraphModel.train_nid = train_ind_sampled
        test_accs, test_f1 = GraphModel.run(epochs=200)
        test_accs_list += test_accs
        test_f1_list += test_f1
        # num_train_sampled += [len(train_ind_sampled)] * (args.epochs-epochs)  


    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)


    np.save(directory + 'GraphSage_accuracy_list_{}_batch_at_scale_sample_method_{}_cold_start{}_redundancy{}_pretrain_ratio{}_ratio{}_total_ratio{}.npy'.format(args.dataset, 
                                                                                                            args.sample_method,
                                                                                                            args.cold_start,
                                                                                                            redundancy,
                                                                                                            args.pretrain_ratio,
                                                                                                            args.ratio,
                                                                                                            args.total_ratio), test_accs_list)

    np.save(directory + 'GraphSage_macro_f1_{}_batch_at_scale_sample_method_{}_cold_start{}_redundancy{}_pretrain_ratio{}_ratio{}_total_ratio{}.npy'.format(args.dataset, 
                                                                                                            args.sample_method,
                                                                                                            args.cold_start,
                                                                                                            redundancy,
                                                                                                            args.pretrain_ratio,
                                                                                                            args.ratio,
                                                                                                            args.total_ratio), test_f1_list)