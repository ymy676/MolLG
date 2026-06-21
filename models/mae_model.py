import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.loss import sce_loss
from itertools import chain
from typing import Optional
from functools import partial

class mae(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            node_decoder: nn.Module,
            num_layers: int,
            dec_in_dim: int,
            cfg,
            node_mask_rate: float = 0.3,
            loss_fn: str = "sce",
            alpha_l: float = 2,
            concat_hidden: bool = False,
         ):
        super().__init__()
        self.cfg = cfg
        self._node_mask_rate = node_mask_rate

        self._concat_hidden = concat_hidden

        self.encoder = encoder
        self.decoder_node = node_decoder
        # build encoder


        self.enc_node_mask_token = nn.Parameter(torch.tensor([119,3],dtype=torch.long),requires_grad=False)
        self.enc_node_remask_token = nn.Parameter(torch.zeros(dec_in_dim), requires_grad=True)

        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.node_criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, g, node_mask_rate=0.3):
        num_nodes = g.num_nodes
        num_mask_nodes = int(node_mask_rate * num_nodes)

        mask_nodes = torch.randperm(num_nodes, device=g.x.device)[:num_mask_nodes]

        new_g = g.clone()          # 关键：复制 graph
        new_g.x = g.x.clone()      # 关键：只复制 x

        new_g.x[mask_nodes] = self.enc_node_mask_token

        return new_g, mask_nodes

    def forward(self, g):
        # ---- attribute reconstruction ----
        loss= self.mask_attr_prediction(g)
        return loss


    def mask_attr_prediction(self, data):

        # =====================================
        # masking
        # =====================================
        x_init = data.x
        masked_data, mask_nodes = self.encoding_mask_noise(
            data,
            self._node_mask_rate
        )

        # =====================================
        # encoder
        # =====================================
        enc_rep, all_hidden = self.encoder(
            masked_data,
            is_mae=True,
        )

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=-1)

        # =====================================
        # decoder input
        # =====================================
        rep = self.encoder_to_decoder(enc_rep)


        rep[mask_nodes] = self.enc_node_remask_token

        # =====================================
        # node reconstruction
        # =====================================
        masked_data.x = rep

        recon_node = self.decoder_node(masked_data, is_mae=True)

        # =====================================
        # loss
        # =====================================
        loss = self.node_criterion(
            recon_node[mask_nodes],
            x_init[mask_nodes]
        )

        return loss

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])