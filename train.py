import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import shutil
import torch
import torch.nn as nn

from utils.seed import set_seed
from utils.logger import get_logger, get_writer
from utils.config import load_config


from data.mydataloader import get_dataloaders

from models.mollg_model import mollg_model
from models.mollg_trainer import mollg_trainer



from models.builder import build_optimizer, build_scheduler
from utils.checkpoint import save_checkpoint, load_checkpoint

from utils import parse_args, create_experiment_dir
def start_from_checkpoint(path):
    checkpoint = torch.load(path, weights_only=False)
    model = checkpoint['model']
    scheduler = checkpoint['scheduler']
    optimizer = checkpoint['optimizer']
    epoch = checkpoint['epoch']
    return model, scheduler, optimizer, epoch
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = create_experiment_dir(cfg)
    shutil.copy(args.config, os.path.join(exp_dir, "config.yaml"))
    logger = get_logger(exp_dir)
    writer = get_writer(exp_dir)
    
    train_loader, val_loader = get_dataloaders(
                                    cfg.dataset.train_path,
                                    cfg.dataset.val_path,
                                    batch_size=cfg.dataset.batch_size,
                                    num_workers=cfg.dataset.num_workers
                                )
    
    model = mollg_model(
        encoder_config=cfg.encoder,
        node_decoder_config=cfg.node_decoder,
        cfg=cfg
    ).to(device)

    #state_dict, _, _, epoch =start_from_checkpoint('outputs/gin_20260614_022048/last_checkpoint.pth')
    #model.load_state_dict(state_dict)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)




    trainer = mollg_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        logger=logger,
        writer=writer, 
        exp_dir=exp_dir,
        cfg=cfg
    )

    trainer.pretrain(train_loader, val_loader, cfg.train.epochs)

    # Evaluate the pre-trained model
    #evaluate_all(trainer.model.encoder, cfg, logger, writer)

if __name__ == "__main__":
    main()