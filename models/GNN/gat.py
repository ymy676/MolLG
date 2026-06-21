import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GATv2Conv
from utils.activation import build_activation
import copy
import torch
num_atom_type = 119 
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class GAT(nn.Module):
    def __init__(self,
                 num_layer,
                 pool,
                 emb_dim,
                 out_dim,
                 drop_ratio,
                 num_heads=4,
                 residual=True,
                 concat=False,
                 activation = None,
                 pretrain=False,
                 encoding=False
                 ):
        super(GAT, self).__init__()
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.gat_layers = nn.ModuleList()
        self.activation = build_activation(activation)
        self.dropout = drop_ratio
        self.pretrain = pretrain
        self.encoding = encoding  
        self.x_embedding1 = nn.Embedding(num_atom_type + 1, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag + 1, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type + 1, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction + 1, emb_dim)
        self.norms = nn.ModuleList()
        for layer in range(num_layer):
            self.norms.append(nn.LayerNorm(emb_dim))
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
        self.conv = GATv2Conv(
                            in_channels=emb_dim,
                            out_channels=emb_dim,
                            heads=num_heads,
                            concat=concat,
                            dropout=0.1,
                            add_self_loops=False,
                            edge_dim=emb_dim,   # 如果你有 bond feature
                            residual=residual,
                            share_weights=False,
                            bias=True
                        )
      
        if num_layer == 1:
            self.gat_layers.append(copy.deepcopy(self.conv))
        else:
            # input projection (no residual)
            self.gat_layers.append(copy.deepcopy(self.conv))
            # hidden layers
            for l in range(1, num_layer - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(copy.deepcopy(self.conv))
            # output projection
            self.gat_layers.append(copy.deepcopy(self.conv))


    def forward(self, data, is_mae=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=torch.long, device=x.device) # (num_nodes, 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])


        if self.encoding:
            h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            h = x
        hidden_list = []
        for l in range(self.num_layer):
            h = self.gat_layers[l](h, edge_index, edge_embeddings)
            if l != self.num_layer - 1:
                h = self.norms[l](h)
            if l != self.num_layer - 1:
                h = F.dropout(self.activation(h), p=self.dropout, training=self.training)
            else:
                h = F.dropout(h, p=self.dropout, training=self.training)

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




