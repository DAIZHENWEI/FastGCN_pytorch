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
import pickle
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.metrics.pairwise import euclidean_distances
from utils import get_batches, accuracy, Entropy_loss
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data, margin_score, dataloader, compute_pagerank, perc, percd, perc_input, percd_input
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


class AnrmabLearner:
    def __init__(self, args, y, normcen, train_index, train_ind_sampled, ix_of_train, device):
        # start_time = time.time()
        super(AnrmabLearner, self).__init__()
        self.device = device

        self.y = y.detach().cpu().numpy()
        # self.NCL = len(np.unique(data.y.cpu().numpy()))
        self.n = len(train_index)
        self.normcen = normcen
        self.w = np.array([1., 1., 1.]) # ie, nc, id
        self.args = args
        self.train_index = train_index
        self.train_ind_sampled = train_ind_sampled
        self.epoch = len(train_ind_sampled)
        self.ix_of_train = ix_of_train
        # print('AnrmabLearner init time', time.time() - start_time)
        
    def pretrain_choose(self, entropy, ed_score, out):
        # here we adopt a slightly different strategy which does not exclude sampled points in previous rounds to keep consistency with other methods
        num_points = self.args.sample_per_round
        scores = entropy
        # epoch = len(self.train_ind_sampled)
        # softmax_out = F.softmax(prev_out, dim=1).cpu().detach().numpy()
        # kmeans = KMeans(n_clusters=self.NCL, random_state=0).fit(softmax_out)
        # ed=euclidean_distances(softmax_out,kmeans.cluster_centers_)
        # ed_score = np.min(ed,axis=1)	#the larger ed_score is, the far that node is away from cluster centers, the less representativeness the node is

        q_ie = scores
        q_nc = self.normcen
        q_id = 1. / (1. + ed_score)
        q_mat = np.vstack([q_ie, q_nc, q_id])  # 3 x n
        q_sum = q_mat.sum(axis=1, keepdims=True)
        q_mat = q_mat / q_sum

        w_len = self.w.shape[0]
        p_min = np.sqrt(np.log(w_len) / w_len / num_points)
        p_mat = (1 - w_len*p_min) * self.w / self.w.sum() + p_min # 3
        
        phi = p_mat[:, np.newaxis] * q_mat # 3 x n
        phi = phi.sum(axis=0) # n

        # sample new points according to phi
        # TODO: change to the sampling method
        # if self.args.anrmab_argmax:
        #     full_new_index_list = np.argsort(phi)[::-1][:num_points] # argmax
        # else:
        #     full_new_index_list = np.random.choice(len(phi), num_points, p=phi)
        train_index_sort = torch.from_numpy(self.train_index[phi.argsort()[::-1]])
        ix = 0
        diff_list = []
        for _ in range(num_points):
            while train_index_sort[ix].item() in self.train_ind_sampled:
                ix += 1
            self.train_ind_sampled = np.append(self.train_ind_sampled, train_index_sort[ix])
            diff_list.append(train_index_sort[ix])

        # mask = combine_new_old(full_new_index_list, self.prev_index_list, num_points, self.n, in_order=True)
        # mask_list = np.where(mask)[0]
        diff_list = self.ix_of_train[np.asarray(diff_list)]
        mask_list = self.ix_of_train[self.train_ind_sampled]
        pred = np.argmax(out, axis=1)
        reward = 1. / num_points / (self.n - num_points) * np.sum((pred[mask_list] == self.y[mask_list]).astype(np.float) / phi[mask_list]) # scalar
        reward_hat = reward * np.sum(q_mat[:, diff_list] / phi[np.newaxis, diff_list], axis=1)
        # update self.w
        # get current node label epoch
        self.epoch += 1
        p_const = np.sqrt(np.log(self.n * 10. / 3. / self.epoch))
        self.w = self.w * np.exp(p_min / 2 * (reward_hat + 1. / p_mat * p_const))

        # import ipdb; ipdb.set_trace()
        # print('Age pretrain_choose time', time.time() - start_time)

        return self.train_ind_sampled


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
    argparser.add_argument('--pre_train_epochs', type=int, default=20, help='Number of pre-training epochs.')
    argparser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--prop-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5,5,5')
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=10)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    argparser.add_argument('--batchsize', type=int, default=256, help='batchsize for train')
    argparser.add_argument('--test_batchsize', type=int, default=2048, help='batchsize for testing')
    argparser.add_argument('--sample_freq', type=int, default=1, help = 'frequnecy of resampling nodes with large uncertainties')
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--seed', type=int, default=123, help='Random seed.')
    argparser.add_argument('--num-workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--pretrain_ratio', type=float, default=0.01, help='Proportion of samples used for training')
    argparser.add_argument('--cluster_per_class', type=int, default=5, help='Number of clusters per class')
    argparser.add_argument('--sample_per_round', type=int, default=1, help='Number of samples per round')
    argparser.add_argument('--ratio', type=float, default=0.01, help='Proportion of samples used for training')
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=1e-4)
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
    labels_train = labels[train_index]
    ix_of_train = -1 * np.ones(len(feats))
    ix_of_train = ix_of_train.astype(int)
    for i, ix in enumerate(np.sort(train_index)):
        ix_of_train[ix] = i

    """ Compute the pagerank centrality """
    pagerank = compute_pagerank(graph)
    train_pagerank = pagerank[train_index]

    # ## Compute the pagerank score
    # if args.dataset not in ['pubmed', 'corafull', 'ogbn-products']:
    #     pagerank_perc = np.asarray([perc(train_pagerank, i) for i in range(len(train_pagerank))])
    # else:
    #     pagerank_perc = perc_input(train_pagerank)

    num_train = len(train_index)
    num_test = len(test_index)
    train_index = train_index.numpy()
    test_index = test_index.numpy()
    n_sampled = int(args.pretrain_ratio * num_train)
    
    train_ind_sampled = np.random.choice(train_index, n_sampled, replace=False)
    train_ind_sampled = torch.from_numpy(train_ind_sampled)
    print("number of sampled nodes: {}:".format(n_sampled))

    labels = labels.to(device)
    feats = feats.to(device)

    data = train_ind_sampled, valid_index, test_index, in_feats, labels, n_classes, feats, graph

    """ Pre-training model """
    test_accs_list, test_f1_list, num_train_sampled = [], [], []
    GraphModel = GraphSage(args, device, data)
    test_accs, test_f1, pred = GraphModel.run(epochs=args.pre_train_epochs, return_embed=True)
    test_accs_list += test_accs
    test_f1_list += test_f1

    NCL = n_classes * args.cluster_per_class

    ANRMAB = AnrmabLearner(args, labels_train, train_pagerank, train_index, train_ind_sampled, ix_of_train, device)

    """ Continue training """
    Entropy = Entropy_loss()
    num_total_sampled = int(num_train * args.ratio)
    epoch = 0
    basef = 0.995
    while len(train_ind_sampled) < num_total_sampled: 
        gamma = np.random.beta(1, 1.005-basef**epoch)
        alpha = beta = (1-gamma)/2
        """ compute entropy """
        entropy_score = Entropy(pred).detach().cpu().numpy()
        train_entropy = entropy_score[train_index]
        # if args.dataset not in ['pubmed', 'corafull', 'ogbn-products']:
        #     entropy_perc = np.asarray([perc(train_entropy,i) for i in range(len(train_entropy))])
        # else:
        #     entropy_perc = perc_input(train_entropy)
        """ compute density score """
        pred = pred.detach().cpu().numpy()
        train_pred = pred[train_index]
        kmeans = KMeans(n_clusters=NCL, random_state=0).fit(train_pred)
        ed=euclidean_distances(train_pred, kmeans.cluster_centers_)
        ed_score = np.min(ed,axis=1)
        # if args.dataset not in ['pubmed', 'corafull', 'ogbn-products']:
        #     ed_perc = np.asarray([percd(ed_score,i) for i in range(len(ed_score))])
        # else:
        #     ed_perc = percd_input(ed_score)

        # finalweight = alpha*entropy_perc + beta*ed_perc + gamma*pagerank_perc

        train_ind_sampled = ANRMAB.pretrain_choose(train_entropy, ed_score, train_pred)
        # train_index_sort = torch.from_numpy(train_index[finalweight.argsort()[::-1]])
        # ix = 0
        # for _ in range(args.sample_per_round):
        #     while train_index_sort[ix].item() in train_ind_sampled:
        #         ix += 1
        #     train_ind_sampled = np.append(train_ind_sampled, train_index_sort[ix])
        print("Number of total selected samples: {}".format(len(train_ind_sampled)))
        GraphModel.train_nid = train_ind_sampled
        test_accs, test_f1, pred = GraphModel.run(epochs=args.sample_freq, return_embed=True)
        test_accs_list += test_accs
        test_f1_list += test_f1
        epoch += 1
    
    test_accs, test_f1 = GraphModel.run(epochs=60)
    test_accs_list += test_accs
    test_f1_list += test_f1

    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)


    np.save(directory + 'GraphSage_accuracy_list_{}_ANRMAB_ratio{}.npy'.format(args.dataset, args.ratio), test_accs_list)

    np.save(directory + 'GraphSage_macro_f1_{}_ANRMAB_ratio{}.npy'.format(args.dataset, args.ratio), test_f1_list)