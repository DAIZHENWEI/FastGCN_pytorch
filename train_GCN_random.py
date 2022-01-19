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
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.metrics import f1_score
from utils import get_batches, accuracy
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data, dataloader
from models import GCN



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
    argparser.add_argument('--fan-out', type=str, default='5,5,5')
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=10)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    argparser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--seed', type=int, default=123, help='Random seed.')
    argparser.add_argument('--num-workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--ratio', type=float, default=0.5,
                        help='Proportion of samples used for training')
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=1e-4)
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


    """ Randomly pick samples"""
    num_sampled = int(len(train_index) * args.ratio)
    print("Number of training nodes: {}".format(len(train_index)))
    train_index = train_index.numpy()
    train_ind_sampled = np.random.choice(train_index, num_sampled, replace=False)
    train_ind_sampled = torch.from_numpy(train_ind_sampled)

    labels = labels.to(device)
    feats = feats.to(device)
    data = train_ind_sampled, valid_index, test_index, in_feats, labels, n_classes, feats, graph, adj_norm



    test_accs_list, test_f1_list, num_train_sampled = [], [], []
    GraphModel = GCN_Model(args, device, data)
    test_accs, test_f1 = GraphModel.run()


    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.remove_degree_one:
        np.save(directory + 'GCN_accuracy_list_{}_random_remove_degree_one_ratio{}.npy'.format(args.dataset, args.ratio), test_accs)
    else:
        np.save(directory + 'GCN_accuracy_list_{}_random_ratio{}.npy'.format(args.dataset, args.ratio), test_accs)


    if args.remove_degree_one:
        np.save(directory + 'GCN_macro_f1_list_{}_random_remove_degree_one_ratio{}.npy'.format(args.dataset, args.ratio), test_f1)
    else:
        np.save(directory + 'GCN_macro_f1_list_{}_random_ratio{}.npy'.format(args.dataset, args.ratio), test_f1)