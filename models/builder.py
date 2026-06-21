from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from models.GNN.gin import GIN
from models.GNN.gcn import GCN
from models.GNN.gat import GAT
import torch.nn as nn
def build_optimizer(cfg, model):

    name = cfg.train.optimizer.lower()

    if name == "adam":
        print(type(cfg.train.lr))
        return Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=1e-5
        )

    elif name == "adamw":
        return AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay
        )

    elif name == "sgd":
        return SGD(
            model.parameters(),
            lr=cfg.train.lr,
            momentum=cfg.train.momentum
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")

def build_scheduler(cfg, optimizer):

    name = cfg.train.scheduler.lower()

    if name == "none":
        return None

    elif name == "step":
        return StepLR(
            optimizer,
            step_size=cfg.train.step_size,
            gamma=cfg.train.gamma
        )

    elif name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.epochs
        )

    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.train.factor,
            patience=cfg.train.patience
        )

    else:
        raise ValueError(f"Unknown scheduler: {name}")
    

def build_model(m_type, emb_dim, out_dim, num_layer, drop_ratio, pool, encoding, pretrain, num_heads=4,residual=False,concat=False) -> nn.Module:
    
    if m_type == "gin":
        mod = GIN(
            emb_dim = emb_dim,
            out_dim = out_dim,
            num_layer=num_layer,
            drop_ratio=drop_ratio,
            pool = pool,
            encoding=encoding,
            pretrain=pretrain,
        )

    elif m_type == "gcn":
        mod = GCN(
            emb_dim = emb_dim,
            out_dim = out_dim,
            num_layer=num_layer,
            drop_ratio=drop_ratio,
            pool = pool,
            encoding=encoding,
            pretrain=pretrain,
        )
    

    elif m_type == "gat":
        mod = GAT(
            emb_dim = emb_dim,
            out_dim = out_dim,
            num_layer=num_layer,
            drop_ratio=drop_ratio,
            pool = pool,
            encoding=encoding,
            pretrain=pretrain,
            num_heads=num_heads,
            residual=residual,
            concat=concat
        )
    
    elif m_type == "mlp":
        assert not encoding, "MLP model only supports decoding mode"
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(out_dim, out_dim*2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim*2, 2)
        )
    elif m_type == "linear":
        assert not encoding, "Linear model only supports decoding mode"
        mod = nn.Linear(out_dim, 2)
    else:
        raise NotImplementedError
    
    return mod

