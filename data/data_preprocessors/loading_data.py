from pathlib import Path
import torch

from torch_geometric.datasets import MoleculeNet

from splitter import random_scaffold_split
from smiles2graph import smiles_to_graph   # 你的函数

ROOT = Path(
    r"E:\Python Projects\egfr_predict\data\finetune"
)


def save_split(train_data, valid_data, test_data, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(train_data, save_dir / "train.pt")
    torch.save(valid_data, save_dir / "val.pt")
    torch.save(test_data, save_dir / "test.pt")

    print(
        f"{save_dir.name}: "
        f"train={len(train_data)}, "
        f"val={len(valid_data)}, "
        f"test={len(test_data)}"
    )


def process_moleculenet(name):

    dataset = MoleculeNet(
        root=f"./datasets/{name}",
        name=name,
    )

    pyg_data_list = []
    smiles_list = []

    for mol in dataset:

        smiles = mol.smiles

        # 用你自己的图构建器
        data = smiles_to_graph(smiles)

        # 保留标签
        data.y = mol.y

        # 保留 smiles（方便以后分析）
        data.smiles = smiles

        pyg_data_list.append(data)
        smiles_list.append(smiles)

    train_data, valid_data, test_data = random_scaffold_split(
        pyg_data_list,
        smiles_list
    )

    save_split(
        train_data,
        valid_data,
        test_data,
        ROOT / name
    )


if __name__ == "__main__":

    datasets = [
        "bace",
        "bbbp",
        "clintox",
        "esol",
        "freesolv",
        "hiv",
        "lipo",
        "muv",
        "sider",
        "tox21",
        "toxcast"
    ]

    for name in datasets:
        print(f"Processing {name}")
        process_moleculenet(name)