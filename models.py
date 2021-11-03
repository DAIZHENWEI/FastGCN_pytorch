import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
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

    def forward(self, x, adj, compute_embed=False):
        outputs = F.relu(self.gc1(x, adj[0]))
        outputs = F.dropout(outputs, self.dropout, training=self.training)
        outputs = F.relu(self.gc2(outputs, adj[1]))
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




# class GCN(nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, sampler):
#         super().__init__()

#         self.gc1 = GraphConvolution(nfeat, nhid)
#         self.gc2 = GraphConvolution(nhid, nclass)
#         self.dropout = dropout
#         self.sampler = sampler
#         self.out_softmax = nn.Softmax(dim=1)

#     def forward(self, x, adj, compute_embed=False):
#         outputs1 = F.relu(self.gc1(x, adj[0]))
#         outputs1 = F.dropout(outputs1, self.dropout, training=self.training)
#         outputs2 = self.gc2(outputs1, adj[1])
#         if compute_embed:
#             return outputs2
#         return F.log_softmax(outputs2, dim=1)


#     def sampling(self, *args, **kwargs):
#         return self.sampler.sampling(*args, **kwargs)
