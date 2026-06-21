from torch_geometric import data
import random
from torch.utils.data import Dataset
import torch

class ContrastiveDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.data_list = data_list
     

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        graph = data["graph"]
        v1, v2 = random.sample(data["views"], 2)

        return graph, v1, v2
        

def load_dataset(path):
    data_list = torch.load(path, weights_only=False)

    dataset = ContrastiveDataset(data_list)

    return dataset


if __name__ == "__main__":
    dataset = load_dataset("data/processed/egfr_graphs/train.pt")
    for i in range(10):
        print(i, type(dataset[i]))