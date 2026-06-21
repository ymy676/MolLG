import pandas as pd
from tqdm import tqdm
import torch
from splitter import random_scaffold_split
from subgraph_removal import subgraph_removal_transform
from smiles2graph import smiles_to_graph as sim2graph
def build_graph_list(csv_path, sim2graph):
    df = pd.read_csv(csv_path)
    smiles_list = df["smiles"].tolist()

    graphs = []
    valid_smiles = []

    for smi in tqdm(smiles_list):
        try:
            g = sim2graph(smi)
            graphs.append(g)
            valid_smiles.append(smi)
        except:
            continue

    return graphs, valid_smiles


from copy import deepcopy

def build_views(graph, subgraph_removal, num_views=20, percent=0.2):
    views = []

    for _ in range(num_views):
        try:
            g = subgraph_removal(deepcopy(graph), percent)
            views.append(g)
        except:
            continue

    return views

from tqdm import tqdm

def build_split_dataset(graphs, subgraph_removal):
    dataset = []

    for g in tqdm(graphs, total=len(graphs)):

        views = build_views(
            g,
            subgraph_removal=subgraph_removal,
            num_views=5,
            percent=0.25
        )

        dataset.append({
            "graph": g,
            "views": views
        })

    return dataset

def save_dataset(dataset, path):
    path = f"E:\\Python Projects\\egfr_predict\\data\\zinc\\{path}"
    torch.save(dataset, path)


def main():
    csv_path = "E:\\Python Projects\\egfr_predict\\data\\zinc250k.csv"

    graphs, smiles = build_graph_list(csv_path, sim2graph)

    train_g, val_g, test_g = random_scaffold_split(graphs, smiles,frac_train=0.9, frac_valid=0.1, frac_test=0.0)


    train_dataset = build_split_dataset(train_g, subgraph_removal_transform)
    val_dataset   = build_split_dataset(val_g, subgraph_removal_transform)

    save_dataset(train_dataset, "train.pt")
    save_dataset(val_dataset, "val.pt")


if __name__ == "__main__":
    main()