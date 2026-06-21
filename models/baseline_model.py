import torch.nn as nn

from models.GNN.gin import GINet
from models.heads import PredHead


class GINRegressionModel(nn.Module):

    def __init__(self, gin_config, head_config):
        super().__init__()

        self.encoder = GINet(**gin_config)
        self.head = PredHead(**head_config)

    def forward(self, data):

        x = self.encoder(data)

        out = self.head(x)

        return out.squeeze(-1)