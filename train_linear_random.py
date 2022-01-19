import argparse
import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import dgl
import torch.nn as nn
import dgl.nn.pytorch as dglnn
import numpy as np
import copy
import pdb
import tqdm
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.metrics import f1_score
from utils import get_batches, accuracy
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data, load_pubmed, load_ogbn_arxiv
from models import SAGE, MLP



def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def macro_f1(pred, labels):
    labels_pred = torch.argmax(pred, dim=1).detach().cpu()
    labels = labels.detach().cpu()
    return f1_score(labels_pred, labels, average='macro')


def evaluate(model, nfeat, labels):
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
        pred = model(nfeat)
    model.train()
    return compute_acc(pred, labels), macro_f1(pred, labels),  pred


#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data
    train_feats = nfeat[train_nid]
    train_labels = labels[train_nid]
    test_feats = nfeat[test_nid]
    test_labels = labels[test_nid]
    # Create PyTorch DataLoader for constructing blocks
    train = data_utils.TensorDataset(train_feats, train_labels)
    dataloader = data_utils.DataLoader(train, batch_size=args.batchsize, shuffle=True)

    # Define model and optimizer
    model = MLP(in_feats, args.hidden, n_classes, args.dropout, args.num_layers)
    # model = nn.Linear(in_feats, n_classes)
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
        for step, (batch_inputs, batch_labels) in enumerate(dataloader):
            tic_step = time.time()

            # Compute loss and prediction
            batch_pred = model(batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(batch_inputs) / (time.time() - tic_step))
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
            test_acc, test_f1, pred = evaluate(model, test_feats, test_labels)
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
    argparser.add_argument('--val-batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=500)
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
    if args.dataset == "ogbn_arxiv":
        loader = load_ogbn_arxiv(args)
    if args.dataset == "pubmed":
        loader = load_pubmed(args)
    train_index, valid_index, test_index, in_feats, labels, n_classes, feats, graph = loader.get_DGL_GCN_inputs()

    """ Compute the feature propagation """
    laplacian = loader.get_norm_laplacian()
    laplacian = sparse_mx_to_torch_sparse_tensor(laplacian).to(device)
    feats = feats.to(device)
    feats_ppg = copy.copy(feats)
    for i in range(args.prop_layers):
        feats_ppg = torch.sparse.mm(laplacian, feats_ppg)
        feats = torch.cat((feats, feats_ppg), dim=1)
    in_feats *= (args.prop_layers + 1)

    # """ Randomly pick the nodes """
    # train_probs = np.random.uniform(0, 1, len(train_index))
    # train_index = train_index.numpy()
    # num_sampled = int(len(train_index) * args.ratio)
    # train_ind_sort = torch.from_numpy(train_index[train_probs.argsort()[::-1]])
    # train_ind_sampled = train_ind_sort[:num_sampled]

    labels = labels.to(device)
    feats = feats.to(device)
    data = train_index, valid_index, test_index, in_feats, labels, n_classes, feats, graph

    # Run 10 times
    test_accs = []
    test_f1 = []
    test_accs, test_f1 = run(args, device, data)


    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if args.remove_degree_one:
        np.save(directory + 'Linear_accuracy_list_{}_random_remove_degree_one_ratio{}.npy'.format(args.dataset, args.ratio), test_accs)
    else:
        np.save(directory + 'Linear_accuracy_list_{}_random_ratio{}.npy'.format(args.dataset, args.ratio), test_accs)


    if args.remove_degree_one:
        np.save(directory + 'Linear_macro_f1_list_{}_random_remove_degree_one_ratio{}.npy'.format(args.dataset, args.ratio), test_f1)
    else:
        np.save(directory + 'Linear_macro_f1_list_{}_random_ratio{}.npy'.format(args.dataset, args.ratio), test_f1)