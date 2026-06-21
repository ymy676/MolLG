from torch_geometric.loader import DataLoader
from data.mydataset import load_dataset

def get_dataloaders(
    train_path,
    val_path,
    batch_size=32,  
    num_workers=2,
):
    train_dataset = load_dataset(train_path)
    val_dataset = load_dataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        #persistent_workers=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        #persistent_workers=True,
        drop_last=True
    )



    return train_loader, val_loader