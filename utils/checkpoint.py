import torch
import os


def save_checkpoint(
    path,
    model,
    optimizer=None,
    logger=None,
    writer=None,
    epoch=None,
    metrics=None,
    scheduler=None
):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics
    }

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()



    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint