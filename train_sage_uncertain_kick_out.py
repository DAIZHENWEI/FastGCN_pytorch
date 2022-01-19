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
from utils import get_batches, accuracy, Entropy_loss
from utils import sparse_mx_to_torch_sparse_tensor
from utils_sage import load_data
from models import SAGE
import pdb


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

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
    return compute_acc(pred[test_nid], labels[test_nid]), pred

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels

#### Entry point
def run(args, model, device, data, epochs, get_embed = False):
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

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    test_acc_list = []
    for epoch in range(epochs):
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
            test_acc, pred = evaluate(model, g, nfeat, labels, val_nid, test_nid, device)
            if args.save_pred:
                np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
            # print('Eval Acc {:.4f}'.format(eval_acc))
            # if eval_acc > best_eval_acc:
            #     best_eval_acc = eval_acc
            #     best_test_acc = test_acc
            print('Test Acc {:.4f}'.format(test_acc))
            test_acc_list += [test_acc.cpu().numpy()]
    # print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    if get_embed:
        return test_acc_list, pred
    return test_acc_list



if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--dataset', type=str, default='cora', help='dataset name.')
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--pre_train_epochs', type=int, default=100, help='Number of pre-training epochs.')
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
    argparser.add_argument('--sample_freq', type=int, default=20, help = 'frequnecy of resampling nodes with large uncertainties')
    argparser.add_argument('--dropout', type=float, default=0.0)
    argparser.add_argument('--seed', type=int, default=123, help='Random seed.')
    argparser.add_argument('--num-workers', type=int, default=4, help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--pretrain_ratio', type=float, default=0.05,
                        help='Proportion of samples used for training')
    argparser.add_argument('--ratio', type=float, default=0.01,
                        help='Proportion of samples picked at each round')
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

    # load ogbn-products data
    train_index, valid_index, test_index, in_feats, labels, n_classes, feats, graph, adj_train, _ = load_data(args.dataset, args)

    # """ Pick the nodes with large laplacian norm"""
    # col_norm = sparse_norm(adj_train, axis=0)
    # train_probs = col_norm / np.sum(col_norm)
    # # train_probs = torch.from_numpy(train_probs)
    # train_index = train_index.numpy()
    # num_sampled = int(len(train_index) * args.ratio)
    # train_ind_sort = torch.from_numpy(train_index[train_probs.argsort()[::-1]])
    # train_ind_sampled = train_ind_sort[:num_sampled]

    # Define model and optimizer
    model = SAGE(in_feats, args.hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


    num_train_sampled = []
    """ randomly pick nodes """
    train_index = train_index.numpy()
    num_sampled = int(len(train_index) * args.pretrain_ratio)
    train_ind_sampled = np.random.choice(train_index, num_sampled, replace = False)

    labels = labels.to(device)
    feats = feats.to(device)
    data = train_ind_sampled, valid_index, test_index, in_feats, labels, n_classes, feats, graph

    """ pretraining model """
    test_accs, pred = run(args, model, device, data, args.pre_train_epochs, get_embed=True)
    test_acc_list = test_accs
    num_train_sampled = [num_sampled] * args.pre_train_epochs

    """ Continue training """
    Entropy = Entropy_loss()
    sample_freq = args.sample_freq
    black_list = np.array([]).reshape(0)
    for epochs in range(0, (args.epochs - args.pre_train_epochs) // sample_freq):
        entropy_score = Entropy(pred).detach().cpu().numpy()
        del pred
        """ Pick nodes with large uncertainty """
        num_sampled = int(len(train_index) * args.ratio)
        train_entropy = entropy_score[train_index]
        train_index_sort = torch.from_numpy(train_index[train_entropy.argsort()[::-1]])
        train_ind_new = train_index_sort[:num_sampled]
        ## newly picked samples already appear in the current training set: need to be kicked out
        train_ind_new_sampled = np.intersect1d(train_ind_new, train_ind_sampled)
        ## update the black list
        train_ind_sample_BL = np.unique(np.concatenate((black_list, train_ind_sampled)))
        black_list = np.unique(np.concatenate((black_list, train_ind_new_sampled)))
        ## newly picked samples that need to be added to the training set
        train_ind_diff = np.setdiff1d(train_ind_sampled, train_ind_new, assume_unique=True)
        train_ind_new_diff = np.setdiff1d(train_ind_new, train_ind_sample_BL, assume_unique=True)    
        train_ind_sampled = np.concatenate((train_ind_diff, train_ind_new_diff))

        print("Number of newly added samples: {}, Number of total selected samples: {}".format(len(train_ind_new_diff), len(train_ind_sampled)))
        print("Number of large entropy nodes need to be kicked out: {}".format(len(train_ind_new_sampled)))
        print("Number of samples in the black list: {}".format(len(black_list)))
        data = train_ind_sampled, valid_index, test_index, in_feats, labels, n_classes, feats, graph
        test_accs, pred = run(args, model, device, data, sample_freq, get_embed=True)
        test_acc_list += test_accs
        num_train_sampled += [len(train_ind_sampled)] * sample_freq


    directory = './save/{}/'.format(args.dataset)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # np.save('./save/GraphSage_accuracy_list_{}_active_laplacian_uncertain_epochs{}_pretrain{}_ratio{}.npy'.format(args.dataset, 
    #                                                                                                     args.epochs,
    #                                                                                                     args.pre_train_epochs, 
    #                                                                                                     args.ratio), test_acc_list)

    np.save(directory + 'GraphSage_accuracy_list_{}_active_random_uncertain_kick_out_pretrain{}_epochs{}_pretrain_ratio{}_ratio{}_sample_freq{}.npy'.format(args.dataset, 
                                                                                                        args.pre_train_epochs, 
                                                                                                        args.epochs,
                                                                                                        args.pretrain_ratio,
                                                                                                        args.ratio,
                                                                                                        args.sample_freq), test_acc_list)

    np.save(directory + 'Sample_count_list_{}_active_random_uncertain_kick_out_pretrain{}_epochs{}_pretrain_ratio{}_ratio{}_sample_freq{}.npy'.format(args.dataset, 
                                                                                                        args.pre_train_epochs, 
                                                                                                        args.epochs,
                                                                                                        args.pretrain_ratio,
                                                                                                        args.ratio,
                                                                                                        args.sample_freq), num_train_sampled)