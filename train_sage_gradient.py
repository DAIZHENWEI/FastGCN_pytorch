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
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.metrics import f1_score
from utils import get_batches, accuracy
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data, load_pubmed, load_ogbn_arxiv
from models import SAGE
import pdb



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

    def run(self, epochs=None):
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
                test_acc, test_f1, _ = self.evaluation()
                test_acc_list += [test_acc.cpu().numpy()]
                test_f1_list += [test_f1]
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
    argparser.add_argument('--pretrain_epochs', type=int, default=10, help='Pre-training epochs before computing the gradients')
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
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--seed', type=int, default=123, help='Random seed.')
    argparser.add_argument('--num-workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--ratio', type=float, default=0.5,
                        help='Proportion of samples used for training')
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--comp_grad', action='store_true', default=False, help="recompute the node gradients")
    argparser.add_argument('--remove_degree_one', action='store_true', default=False,
                        help='Recursively remove the nodes with degree one from the adjacency matrix (remove corresponding edges).')
    args = argparser.parse_args()
    

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    if args.gpu != -1:
        torch.cuda.manual_seed(args.seed)

    if args.gpu >= 0:
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    # load ogbn-products data
    """ Load dataset """
    if args.dataset == "ogbn_arxiv":
        loader = load_ogbn_arxiv(args)
    if args.dataset == "pubmed":
        loader = load_pubmed(args)
    train_index, valid_index, test_index, in_feats, labels, n_classes, feats, graph = loader.get_DGL_GCN_inputs()

    labels = labels.to(device)
    feats = feats.to(device)
    data = train_index, valid_index, test_index, in_feats, labels, n_classes, feats, graph

    """ Compute the gradient of each node """
    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_dir = directory + 'GraphSage_node_gradients_avg_{}_pretrain{}_hidden{}_layer{}.npy'.format(args.dataset, args.pretrain_epochs, args.hidden, args.num_layers)
    
    if not os.path.exists(file_dir) or args.comp_grad:
        grad_norm_avg = []
        grad_norms = []
        num_rep = 10
        for i in range(num_rep):
            GraphModel = GraphSage(args, device, data)
            GraphModel.run(epochs = args.pretrain_epochs)
            grad_norm = GraphModel.compute_gradient()
            if not grad_norm_avg:
                grad_norm_avg = [norm/num_rep for norm in grad_norm]
                grad_norms = grad_norm[np.newaxis,:]
            else:
                grad_norm_avg = [norm/num_rep+norm_avg for norm, norm_avg in zip(grad_norm, grad_norm_avg)]
                grad_norms = np.concatenate((grad_norms, grad_norm[np.newaxis,:]))
        grad_norm_avg = np.array(grad_norm_avg)

        if num_rep > 1:
            np.save(directory + 'GraphSage_node_gradients_avg_{}_pretrain{}_hidden{}_layer{}.npy'.format(args.dataset, args.pretrain_epochs, args.hidden, args.num_layers), grad_norm_avg)
            np.save(directory + 'GraphSage_node_gradients_all_{}_pretrain{}_hidden{}_layer{}.npy'.format(args.dataset, args.pretrain_epochs, args.hidden, args.num_layers), grad_norms)
    else:
        grad_norm_avg = np.load(file_dir)


    """ Pick the nodes with large gradients """
    train_index = train_index.numpy()
    num_sampled = int(len(train_index) * args.ratio)
    train_ind_sort = torch.from_numpy(train_index[grad_norm_avg.argsort()[::-1]])
    train_ind_sampled = train_ind_sort[:num_sampled]

    GraphModel = GraphSage(args, device, data)
    GraphModel.train_nid = train_ind_sampled 
    test_accs, test_f1 = GraphModel.run()

    if args.remove_degree_one:
        np.save(directory + 'GraphSage_accuracy_list_{}_large_gradient_pretrain{}_remove_degree_one_ratio{}.npy'.format(args.dataset, args.pretrain_epochs, args.ratio), test_accs)
    else:
        np.save(directory + 'GraphSage_accuracy_list_{}_large_gradient_pretrain{}_ratio{}.npy'.format(args.dataset, args.pretrain_epochs, args.ratio), test_accs)


    if args.remove_degree_one:
        np.save(directory + 'GraphSage_macro_f1_list_{}_large_gradient_pretrain{}_remove_degree_one_ratio{}.npy'.format(args.dataset, args.pretrain_epochs, args.ratio), test_f1)
    else:
        np.save(directory + 'GraphSage_macro_f1_list_{}_large_gradient_pretrain{}_ratio{}.npy'.format(args.dataset, args.pretrain_epochs, args.ratio), test_f1)