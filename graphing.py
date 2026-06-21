import os
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from utils.config import load_config
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from models.mollg_model import mollg_model
path = 'outputs/gin_20260612_025604'
def get_loaders(root, name, batch_size=512):
    base = f"{root}/{name}"

    train_ds = MoleculeSplitDataset(f"{base}/train.pt")
    val_ds   = MoleculeSplitDataset(f"{base}/val.pt")
    test_ds  = MoleculeSplitDataset(f"{base}/test.pt")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class MoleculeSplitDataset(Dataset): 
    def __init__(self, path): 
        self.data_list = torch.load(path, weights_only=False) 
    def __len__(self): 
        return len(self.data_list) 
    def __getitem__(self, idx): 
        return self.data_list[idx]



single_tasks = {
    "bbbp": "classification",
    "bace": "classification",
    "hiv": "classification",
    "esol": "regression",
    "freesolv": "regression",
    "lipo": "regression"
}

plot_config = {

    "bbbp": {
        "title": "BBBP",
        "legend": {
            0: ("Non-BBB permeable", "#4C72B0", "o"),
            1: ("BBB permeable", "#DD8452", "^")
        }
    },

    "bace": {
        "title": "BACE",
        "legend": {
            0: ("Inactive", "#4C72B0", "o"),
            1: ("Active", "#DD8452", "^")
        }
    },

    "hiv": {
        "title": "HIV",
        "legend": {
            0: ("Inactive", "#4C72B0", "o"),
            1: ("Active", "#DD8452", "^")
        }
    },

    "esol": {
        "title": "ESOL",
        "colorbar_label": "Solubility"
    },

    "freesolv": {
        "title": "FreeSolv",
        "colorbar_label": "Hydration Free Energy"
    },

    "lipo": {
        "title": "Lipophilicity",
        "colorbar_label": "Experimental LogD"
    }
}


def extract_latent(encoder, loaders, device):
    encoder.eval()

    reps = []
    labels = []

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                batch = batch.to(device)

                # ===== 根据自己的 encoder 修改 =====
                z,_,_ = encoder(batch)

                # 如果 encoder 返回 tuple:
                # _, _, z = encoder(batch)

                reps.append(z.cpu())
                labels.append(batch.y.cpu())

    reps = torch.cat(reps, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy().reshape(-1)

    return reps, labels


def draw_tsne(reps, labels, task_name, task_type):

    cfg = plot_config[task_name]

    reps = StandardScaler().fit_transform(reps)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init="pca"
    )

    emb = tsne.fit_transform(reps)

    plt.figure(figsize=(7,6))

    if task_type == "classification":

        labels = labels.astype(int)

        for cls, (name, color, marker) in cfg["legend"].items():

            idx = labels == cls

            plt.scatter(
                emb[idx,0],
                emb[idx,1],
                c=color,
                marker=marker,
                s=20,
                alpha=0.75,
                label=name
            )

        plt.legend(
            fontsize=11,
            frameon=True,
            edgecolor="black"
        )

    else:

        sc = plt.scatter(
            emb[:,0],
            emb[:,1],
            c=labels,
            cmap="viridis",
            s=20,
            alpha=0.7
        )

        cbar = plt.colorbar(sc)
        cbar.set_label(
            cfg["colorbar_label"],
            fontsize=12
        )

    plt.title(
        f"{cfg['title']} latent space visualization",
        fontsize=15
    )

    plt.xlabel("t-SNE dimension 1", fontsize=12)
    plt.ylabel("t-SNE dimension 2", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    os.makedirs("tsne_figures", exist_ok=True)

    plt.savefig(
        f"tsne_figures/{task_name}_tsne.png",
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"{task_name} finished.")


if __name__ == "__main__":

    checkpoint = torch.load(
        os.path.join(path, "best_probe_checkpoint.pth"),
        map_location="cpu"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    cfg = load_config(f'{path}/config.yaml')
    model = mollg_model(
        encoder_config=cfg.encoder,
        node_decoder_config=cfg.node_decoder,
        cfg=cfg
    ).to(device)

    model.load_state_dict(checkpoint["model"])

    encoder = model.encoder

    for task_name, task_type in single_tasks.items():

        print(f"Processing {task_name}...")

        train_loader, val_loader, test_loader = get_loaders(
            root="data/finetune",
            name=task_name,
            batch_size=32
        )

        reps, labels = extract_latent(
            encoder,
            [train_loader, val_loader, test_loader],
            device
        )

        draw_tsne(
            reps,
            labels,
            task_name,
            task_type
        )