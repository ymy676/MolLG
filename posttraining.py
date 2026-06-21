import torch
import json

from models.mollg_model import mollg_model
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from finetune import evaluate_all
from utils.config import load_config
from utils.logger import get_logger, get_writer
from itertools import product
def get_logger(save_dir):

    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s"
    )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    # file
    fh = logging.FileHandler(
        os.path.join(save_dir, "test.log")
    )
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

def get_writer(save_dir):
    save_dir = os.path.join(save_dir, "tensorboard")
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    return writer

paths=["gin_20260607_183933",
        "gin_20260612_025604",
        "gin_20260613_011221",
        "gin_20260614_022048",
        "gin_20260615_014710",
        "gin_20260615_083150",
        "gin_20260616_020430"]

search_space = product(
    [1e-4, 5e-4, 5e-5],       # lr
    [0, 1e-6, 1e-5],          # weight_decay
    [0.1, 0.3, 0.5],          # dropout
    [32, 64]                  # batch_size
)

all_results = {}

if __name__ == "__main__":
    for path in paths:
        best_improvement = float('-inf')
        for lr, wd, drop, bs in search_space:
            finetune_config = {"lr": lr, "dropout": drop, "weight_decay": wd, "batch_size": bs}
            dir_path = f"outputs/{path}"
            cfg = load_config("configs/gin.yaml")
            logger = get_logger(dir_path)
            writer = get_writer(dir_path)
            checkpoint = torch.load(os.path.join(dir_path, "best_probe_checkpoint.pth"), map_location="cpu")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = mollg_model(
                encoder_config=cfg.encoder,
                node_decoder_config=cfg.node_decoder,
                cfg=cfg
            ).to(device)
            model.load_state_dict(checkpoint["model"])
            encoder = model.encoder
            results, improvement = evaluate_all(encoder, cfg, logger, writer,lr,wd,drop,bs,task1=True)
            if improvement > best_improvement:
                best_param = (lr,drop,wd,bs)
            finetune_config = {"lr": lr, "dropout": drop, "weight_decay": wd, "batch_size": bs}

#######HIV,MUV#######
        lr,drop,wd,bs = best_param
        checkpoint = torch.load(os.path.join(dir_path, "best_probe_checkpoint.pth"), map_location="cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = mollg_model(
                encoder_config=cfg.encoder,
                node_decoder_config=cfg.node_decoder,
                cfg=cfg
            ).to(device)
        model.load_state_dict(checkpoint["model"])
        encoder = model.encoder
        results, improvement = evaluate_all(encoder, cfg, logger, writer,lr,wd,drop,bs,task1=False)
         
