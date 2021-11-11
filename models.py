import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import dgl
import dgl.nn.pytorch as dglnn
import tqdm
import pdb


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.out = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj, compute_embed=False, test=False):
        outputs = F.relu(self.gc1(x, adj[0]))
        outputs = F.dropout(outputs, self.dropout, training=self.training)
        outputs = F.relu(self.gc2(outputs, adj[1]))
        if compute_embed:
            return outputs
        outputs = self.out(outputs)
        return F.log_softmax(outputs, dim=1)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)



class GCN3(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.out = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.layernorm = nn.LayerNorm(nhid)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj, compute_embed=False, test=False):
        if not test:
            outputs = self.gc1(x, adj[0])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc2(outputs, adj[1])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc3(outputs, adj[2])
            outputs = F.relu(self.layernorm(outputs))
        else:
            outputs = self.gc1(x, adj[0])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc2(outputs, adj[0])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc3(outputs, adj[1])
            outputs = F.relu(self.layernorm(outputs))          
        if compute_embed:
            return outputs
        outputs = self.out(outputs)
        return F.log_softmax(outputs, dim=1)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)


class GCN4(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, sampler):
        super().__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nhid)
        self.out = nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.sampler = sampler
        self.layernorm = nn.LayerNorm(nhid)
        self.out_softmax = nn.Softmax(dim=1)

    def forward(self, x, adj, compute_embed=False, test=False):
        if not test:
            outputs = self.gc1(x, adj[0])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc2(outputs, adj[1])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc3(outputs, adj[2])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc4(outputs, adj[3])
            outputs = F.relu(self.layernorm(outputs))
        else:
            outputs = self.gc1(x, adj[0])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc2(outputs, adj[0])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc3(outputs, adj[0])
            outputs = F.relu(self.layernorm(outputs))
            outputs = F.dropout(outputs, self.dropout, training=self.training)
            outputs = self.gc4(outputs, adj[1])
            outputs = F.relu(self.layernorm(outputs))          
        if compute_embed:
            return outputs
        outputs = self.out(outputs)
        return F.log_softmax(outputs, dim=1)

    def sampling(self, *args, **kwargs):
        return self.sampler.sampling(*args, **kwargs)




class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation


    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


    def inference(self, g, x, device, args):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = torch.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                torch.arange(g.num_nodes()),
                sampler,
                batch_size=args.batchsize,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes]
                h_dst = h[:block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y
        return y
