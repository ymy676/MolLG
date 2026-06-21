import torch
import torch.nn as nn
from models.clr_model import CLR
from models.builder import build_model
from models.mae_model import mae

import math

def get_gamma_scheduler(
    schedule_type="warmup_const",
    max_lambda=0.1,
    warmup_epochs=30,
    total_epochs=100
):
    """
    return: function(epoch) -> lambda
    """

    # -------------------------
    # 1) warmup + constant (推荐默认)
    # -------------------------
    if schedule_type == "warmup_const":
        def fn(epoch):
            if epoch < warmup_epochs:
                return max_lambda * epoch / warmup_epochs
            return max_lambda
        return fn

    # -------------------------
    # 2) cosine schedule
    # -------------------------
    elif schedule_type == "cosine":
        def fn(epoch):
            return max_lambda * 0.5 * (
                1 + math.cos(math.pi * epoch / total_epochs)
            )
        return fn

    # -------------------------
    # 3) step schedule
    # -------------------------
    elif schedule_type == "step":
        def fn(epoch):
            if epoch < warmup_epochs:
                return 0.0
            elif epoch < warmup_epochs * 2:
                return max_lambda * 0.5
            else:
                return max_lambda
        return fn

    # -------------------------
    # 4) linear decay (不太推荐用于你任务)
    # -------------------------
    elif schedule_type == "linear_decay":
        def fn(epoch):
            return max_lambda * (1 - epoch / total_epochs)
        return fn

    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")
    
class mollg_model(nn.Module):
    def __init__(
            self,
            encoder_config,
            node_decoder_config,
            cfg,
        ):
        super().__init__()

        self.encoder = build_model(**encoder_config)
        self.node_decoder = build_model(**node_decoder_config)
        if cfg.train.gamma_scheduler:
            self.gamma_scheduler = get_gamma_scheduler(
                schedule_type=cfg.train.gamma_scheduler,
                max_lambda=cfg.model.gamma,
                warmup_epochs=cfg.train.gamma_warmup_epochs,
                total_epochs=cfg.train.epochs
            )
        self.mae_model = mae(encoder=self.encoder,
                            node_decoder=self.node_decoder,
                            cfg=cfg,
                            **cfg.mae)
        self.clr_model = CLR(encoder=self.encoder,cfg=cfg)

        self.gamma = cfg.model.gamma
        self.cfg = cfg

    def forward(self, g, xis, xjs, gamma=None, epoch=None):
        clr_loss = self.clr_model(xis, xjs)
        mae_loss = self.mae_model(g)
        if gamma is None:
            if self.cfg.train.gamma_scheduler and epoch is not None:
                gamma = self.gamma_scheduler(epoch)
            else:
                gamma = self.gamma
        loss = mae_loss + gamma * clr_loss
        return loss, {'mae_loss': mae_loss.item(), 'clr_loss': clr_loss.item()}, gamma

    
