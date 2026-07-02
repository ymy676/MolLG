import torch
import json

from models.mollg_model import mollg_model
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from utils.finetuner import evaluate_all
from utils.config import load_config
from itertools import product


def get_logger(save_dir):

    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger(f"train_{save_dir}")
    logger.setLevel(logging.INFO)

    # 防止重复 handler（很重要，不然循环会重复写日志）
    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s - %(message)s"
        )

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)

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


paths = [
    #"gin_20260607_183933",
    #"gin_20260612_025604",
    #"gin_20260614_022048",
    "gin_20260615_014710",
    "gin_20260615_083150",
    "gin_20260616_020430"
]

# ✅ 必须转 list，否则 iterator 会被消耗
search_space = list(product(
    [1e-4],       # lr
    [1e-5],          # weight_decay
    [0.1],          # dropout
    [32]                  # batch_size
))

all_results = {}

if __name__ == "__main__":

    for path in paths:

        best_improvement = float('-inf')
        best_param = None

        dir_path = f"outputs/{path}"
        os.makedirs(dir_path, exist_ok=True)

        # ✅ 每个 path 单独存 grid search 结果
        result_file = os.path.join(dir_path, "grid_search_results.jsonl")
        best_file = os.path.join(dir_path, "best_param.json")

        cfg = load_config("configs/gin.yaml")

        logger = get_logger(dir_path)
        writer = get_writer(dir_path)

        checkpoint = torch.load(
            os.path.join(dir_path, "best_probe_checkpoint.pth"),
            map_location="cpu"
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = mollg_model(
            encoder_config=cfg.encoder,
            node_decoder_config=cfg.node_decoder,
            cfg=cfg
        ).to(device)

        model.load_state_dict(checkpoint["model"])
        encoder = model.encoder

        # =========================
        # GRID SEARCH
        # =========================
        for lr, wd, drop, bs in search_space:

            finetune_config = {
                "lr": lr,
                "dropout": drop,
                "weight_decay": wd,
                "batch_size": bs
            }

            results, improvement = evaluate_all(
                encoder, cfg, logger, writer,
                lr, wd, drop, bs,
                task1=True
            )

            # ======= 1. 实时写 JSONL（防中断核心） =======
            record = {
                "lr": lr,
                "wd": wd,
                "dropout": drop,
                "batch_size": bs,
                "improvement": float(improvement),
                "results": results
            }

            with open(result_file, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()

            # ======= 2. 更新 best =======
            if improvement > best_improvement:
                best_improvement = improvement
                best_param = (lr, drop, wd, bs)

                # 实时保存 best
                with open(best_file, "w") as f:
                    json.dump({
                        "best_improvement": float(best_improvement),
                        "best_param": {
                            "lr": lr,
                            "dropout": drop,
                            "weight_decay": wd,
                            "batch_size": bs
                        }
                    }, f, indent=2)

        # =========================
        # 用 best param 跑 HIV / MUV
        # =========================
        """
        lr, drop, wd, bs = best_param

        checkpoint = torch.load(
            os.path.join(dir_path, "best_probe_checkpoint.pth"),
            map_location="cpu"
        )

        model = mollg_model(
            encoder_config=cfg.encoder,
            node_decoder_config=cfg.node_decoder,
            cfg=cfg
        ).to(device)

        model.load_state_dict(checkpoint["model"])
        encoder = model.encoder

        results, improvement = evaluate_all(
            encoder, cfg, logger, writer,
            lr, wd, drop, bs,
            task1=False
        )

        # final 保存
        with open(os.path.join(dir_path, "final_result.json"), "w") as f:
            json.dump({
                "best_param": {
                    "lr": lr,
                    "dropout": drop,
                    "weight_decay": wd,
                    "batch_size": bs
                },
                "final_improvement": float(improvement),
                "final_results": results
            }, f, indent=2)
            """