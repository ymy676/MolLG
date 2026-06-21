import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv

num_atom_type = 119 
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class GCN(nn.Module):
    def __init__(self,
                 pool,
                 emb_dim,
                 out_dim,
                 num_layer,
                 drop_ratio,
                 activation = None,
                 pretrain=False,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = drop_ratio
        self.pretrain = pretrain
        self.encoding = encoding  
        self.x_embedding1 = nn.Embedding(num_atom_type + 1, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag + 1, emb_dim)
        self.norms = nn.ModuleList()
        for layer in range(num_layer):
            self.norms.append(nn.BatchNorm1d(emb_dim))
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.classifier = nn.Linear(self.emb_dim, 2)
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.out_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim*2), 
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim*2, self.out_dim)
        )
        
      
        if num_layer == 1:
            self.gcn_layers.append(GCNConv(in_channels=emb_dim, out_channels=emb_dim, add_self_loops=True, normalize=True, bias=True))
        else:
            # input projection (no residual)
            self.gcn_layers.append(GCNConv(in_channels=emb_dim, out_channels=emb_dim, add_self_loops=True, normalize=True, bias=True))
            # hidden layers
            for l in range(1, num_layer - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gcn_layers.append(GCNConv(in_channels=emb_dim, out_channels=emb_dim, add_self_loops=True, normalize=True, bias=True))
            # output projection
            self.gcn_layers.append(GCNConv(in_channels=emb_dim, out_channels=emb_dim, add_self_loops=True, normalize=True, bias=True))


    def forward(self, data, is_mae=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if self.encoding:
            h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            h = x
        hidden_list = []
        for l in range(self.num_layer):
            if l != self.num_layer - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](h, edge_index)
            if l != self.num_layer - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        if is_mae:
            if not self.encoding:
                h = self.classifier(h)
                return h
            return h, hidden_list
        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        if self.pretrain:
            out = self.out_lin(h)
            return h, out, hidden_list
        return h, hidden_list

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)
