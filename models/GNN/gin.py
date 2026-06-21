import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


num_atom_type = 119 
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3 

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type + 1, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction + 1, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=torch.long, device=x.device) # (num_nodes, 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GIN(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, encoding=False, emb_dim=300, out_dim=256, drop_ratio=0, pool='mean', pretrain=False):
        super(GIN, self).__init__()
        self.encoding = encoding
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.drop_ratio = drop_ratio
        self.pretrain = pretrain
        self.x_embedding1 = nn.Embedding(num_atom_type + 1, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag + 1, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.classifier = nn.Linear(self.emb_dim, 2)
        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
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

    def forward(self, data, is_mae=False):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if self.encoding:
            h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])
        else:
            h = x
        hidden_list = []
        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            hidden_list.append(h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
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