import logging
import os
from torch.utils.tensorboard import SummaryWriter

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
        os.path.join(save_dir, "train.log")
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

