import torch.nn as nn
from utils.loss import NTXentLoss
import torch.nn.functional as F



class CLR(nn.Module):
    def __init__(self, encoder,cfg):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.nt_xent_criterion = NTXentLoss(self.cfg.dataset.batch_size, self.cfg.clr.temperature, self.cfg.clr.use_cosine_similarity)

    def forward(self, xis, xjs):
        # get the representations and the projections
        zis, proj_zis, hidden_list = self.encoder(xis)  # [N,C]

        # get the representations and the projections
        zjs, proj_zjs, hidden_list = self.encoder(xjs)  # [N,C]

        # normalize projection feature vectors
        proj_zis = F.normalize(proj_zis, dim=1)
        proj_zjs = F.normalize(proj_zjs, dim=1)

        loss = self.nt_xent_criterion(proj_zis, proj_zjs)
        return loss






