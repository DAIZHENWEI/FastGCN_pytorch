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
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import copy
import pdb
import tqdm
from scipy.sparse.linalg import norm as sparse_norm
from utils import get_batches, accuracy
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data, dataloader
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

#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)
    # Define model and optimizer
    model = SAGE(in_feats, args.hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    test_acc_list = []
    test_f1_list = []
    for epoch in range(args.epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(nfeat, labels, seeds, input_nodes)
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
        
        # print('Number of steps per epochs: {}'.format(step+1))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0:
            test_acc, test_f1, pred = evaluate(model, g, nfeat, labels, val_nid, test_nid, device)
            if args.save_pred:
                np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            # print('Eval Acc {:.4f}'.format(eval_acc))
            # if eval_acc > best_eval_acc:
            #     best_eval_acc = eval_acc
            #     best_test_acc = test_acc
            print('Test Acc {:.4f}'.format(test_acc))
            print('Test Macro F1 {:.4f}'.format(test_f1))
            test_acc_list += [test_acc.cpu().numpy()]
            test_f1_list += [test_f1]
    # print('Avg epoch time: {}'.format(avg / (epoch - 4)))
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
    argparser.add_argument('--test_batchsize', type=int, default=2048, help='batchsize for testing')
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
    laplacian = sparse_mx_to_torch_sparse_tensor(laplacian).to(device)
    feats_ppg = copy.copy(feats)
    feats_ppg = feats_ppg.to(device)
    for i in range(args.prop_layers):
        feats_ppg = torch.sparse.mm(laplacian, feats_ppg)
    del laplacian
    torch.cuda.empty_cache()


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

    if prop_to_sd:
        dists = []
        for i in range(n_cluster):
            feats_i = feats_train[clusters == i]
            dists.append(np.sum(LA.norm(feats_i - centers[i], axis = 1, ord=1)))
        cluster_size_sample = [dist*num_train*args.ratio/np.sum(dists) for dist in dists]
    else:
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
    data = train_ind_sampled, valid_index, test_index, in_feats, labels, n_classes, feats, graph

    # Run 10 times
    test_accs = []
    test_f1 = []
    test_accs, test_f1 = run(args, device, data)


    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)


    np.save(directory + 'GraphSage_accuracy_list_{}_Kmedoid_ncluster{}_PropToSd{}_AggregateLayer{}_ratio{}.npy'.format(args.dataset, 
                                                                                                            n_cluster,
                                                                                                            args.prop_to_sd,
                                                                                                            args.prop_layers, 
                                                                                                            args.ratio), test_accs)

    np.save(directory + 'GraphSage_micro_f1_list_{}_Kmedoid_ncluster{}_PropToSd{}_AggregateLayer{}_ratio{}.npy'.format(args.dataset, 
                                                                                                            n_cluster,
                                                                                                            args.prop_to_sd,
                                                                                                            args.prop_layers, 
                                                                                                            args.ratio), test_f1)





