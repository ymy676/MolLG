from sympy import re
import torch
import torch.nn as nn
from utils.seed import set_seed
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import copy
from torch_geometric.nn import global_max_pool
from utils.loss import MaskedBCEWithLogitsLoss
# --- Dataset and Device Setup ---

class MoleculeSplitDataset(Dataset): 
    def __init__(self, path): 
        self.data_list = torch.load(path, weights_only=False) 
    def __len__(self): 
        return len(self.data_list) 
    def __getitem__(self, idx): 
        return self.data_list[idx]
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Then in get_loss:
    

def get_loaders(root, name, batch_size=512):
    base = f"{root}/{name}"

    train_ds = MoleculeSplitDataset(f"{base}/train.pt")
    val_ds   = MoleculeSplitDataset(f"{base}/val.pt")
    test_ds  = MoleculeSplitDataset(f"{base}/test.pt")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# --- Model Architecture ---

class GNNFinetuneModel(nn.Module):
    def __init__(self, encoder, hidden_dim, task_type,dropout, num_tasks=1):
        super().__init__()
        self.encoder = encoder
        #self.encoder.pool = global_max_pool

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks)
        )
        self.task_type = task_type

    def forward(self, data):
        # Unpack your encoder's outputs (assuming it returns node embeddings first)
        h, _, _ = self.encoder(data)
        
        # If your encoder doesn't pool automatically, apply pooling here:
        # if h.size(0) != data.num_graphs:
        #     h = self.pool(h, data.batch)
        
        out = self.head(h)
        return out

# --- Loss Function Router ---

def get_loss(task_type):
    if task_type in ["binary"]:
        # "none" reduction is critical for row/column-wise NaN masking in train()
        return nn.BCEWithLogitsLoss()
    elif task_type in ["multitask_binary"]:
        return MaskedBCEWithLogitsLoss()
    elif task_type == "regression":
        return nn.MSELoss()

# --- Evaluation Pipeline ---

def evaluate(task_type, y_true, y_pred):
    # Ensure inputs are clean numpy arrays for reliable indexing
    if hasattr(y_true, 'cpu'): y_true = y_true.cpu().numpy()
    if hasattr(y_pred, 'cpu'): y_pred = y_pred.cpu().numpy()

    if task_type == "binary":
        return roc_auc_score(y_true, y_pred)

    elif task_type == "multitask_binary":
        # Force 2D scaling matrix shapes [Samples, Tasks]
        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        num_tasks = y_true.shape[1]
        task_auc_scores = []

        for task in range(num_tasks):
            t_true = y_true[:, task]
            t_pred = y_pred[:, task]

            # Jointly strip out NaNs from targets and any corrupted values in preds
            valid_indices = (
                ~np.isnan(t_true) & 
                ~np.isnan(t_pred) & 
                np.isfinite(t_pred)
            )
            t_true_clean = t_true[valid_indices]
            t_pred_clean = t_pred[valid_indices]

            # Skip the task if both classes (0 and 1) aren't present in this validation slice
            if len(np.unique(t_true_clean)) < 2:
                continue

            auc = roc_auc_score(t_true_clean, t_pred_clean)
            task_auc_scores.append(auc)

        # Return macro-average AUC across all validly evaluated tasks
        return np.mean(task_auc_scores) if len(task_auc_scores) > 0 else 0.5

    elif task_type == "regression":
        return root_mean_squared_error(y_true, y_pred)

# --- Train and Test Steps ---

def train(model, loader, optimizer, loss_fn):

    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)


        pred = model(data)

        y = data.y.float()


        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def test(model, loader, task_type):
    model.eval()
    ys, preds = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)

            ys.append(data.y.cpu())
            preds.append(pred.cpu())

    y = torch.cat(ys).numpy()
    p = torch.cat(preds).numpy()

    return evaluate(task_type, y, p)

# --- Global Orchestration Configuration ---

TASK_CONFIG = {
    #"muv":        {"task": "multitask_binary", "metric": "rocauc"},
    "bbbp":       {"task": "binary", "metric": "rocauc"},
    "bace":       {"task": "binary", "metric": "rocauc"},
    #"hiv":        {"task": "binary", "metric": "rocauc"},
    "clintox":    {"task": "binary", "metric": "rocauc"},

    
    "tox21":      {"task": "multitask_binary", "metric": "rocauc"},
    "toxcast":    {"task": "multitask_binary", "metric": "rocauc"},
    "sider":      {"task": "multitask_binary", "metric": "rocauc"},

    "esol":       {"task": "regression", "metric": "rmse"},
    "freesolv":   {"task": "regression", "metric": "rmse"},
    "lipo":       {"task": "regression", "metric": "rmse"},
}

TASK_CONFIG2 = {"muv":        {"task": "multitask_binary", "metric": "rocauc"},
                "hiv":        {"task": "binary", "metric": "rocauc"}}
molclr_baseline = {
    "BBBP": 73.3,
    "Tox21": 74.1,
    "ToxCast": 65.9,
    "SIDER": 61.2,
    "ClinTox": 89.8,
    "BACE": 82.8,
    "MUV": 78.9,
    "HIV": 77.4,
    "ESOL": 1.113,
    "FreeSolv": 2.301,
    "Lipo": 0.789,
}
def evaluate_all(encoder, config, logger, writer, lr, weight_decay, dropout, batchsize, task1 = True):
    epochs = config.eval.epochs
    results = {}
    my_task = TASK_CONFIG if (task1 == True) else TASK_CONFIG2
    all_improvement = 0
    for name, cfg in my_task.items():
        results[name] = {}
        
        for trial in range(3):
            set_seed(trial*5)

            logger.info(f"\n===== Running {name} (Trial {trial + 1}) =====")
            if cfg["metric"] == "rocauc":
                best_score = float('-inf')
            elif cfg["metric"] == "rmse":
                best_score = float('inf')
            train_loader, val_loader, test_loader = get_loaders(root="data/finetune", name=name, batch_size=batchsize)
            data = train_loader.dataset[0]  
            my_encoder = copy.deepcopy(encoder)
            model = GNNFinetuneModel(
                encoder=my_encoder,
                hidden_dim=config.encoder.out_dim,
                task_type=cfg["task"],
                dropout=dropout,
                num_tasks=data.y.shape[-1] if len(data.y.shape) > 1 else 1
            ).to(device)
            #All freeze
            for p in model.encoder.parameters():
                p.requires_grad = False

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            #optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.9,nesterov=True,weight_decay=1e-4)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            loss_fn = get_loss(cfg["task"])

            for epoch in range(epochs):
                # -------- stage switching --------
                if epoch == int(0.5 * epochs):
                    logger.info("Unfreezing last encoder layer")

                    for p in model.encoder.gnns[-1].parameters():
                        p.requires_grad = True
                    for p in model.encoder.batch_norms[-1].parameters():
                        p.requires_grad = True

                    # encoder lr smaller (IMPORTANT)
                    for g in optimizer.param_groups:
                        if g["lr"] == 1e-4:
                            g["lr"] = 3e-5

                if epoch == int(0.8 * epochs):
                    logger.info("Full encoder fine-tuning")

                    for p in model.encoder.parameters():
                        p.requires_grad = True

                    for g in optimizer.param_groups:
                        g["lr"] = 1e-5
                            
                loss = train(model, train_loader, optimizer, loss_fn)
                val_score = test(model, val_loader, cfg["task"])
                logger.info(f"Epoch {epoch+1:02d}, Loss: {loss:.4f}, Val {cfg['metric']}: {val_score:.4f}")
                writer.add_scalar(f"trial:{trial + 1}_{name}_{cfg['metric']}", val_score, epoch)
                #scheduler.step()
                if (cfg["metric"] == "rocauc" and val_score > best_score) or (cfg["metric"] == "rmse" and val_score < best_score):
                    best_score = val_score
                    best_state = copy.deepcopy(model.state_dict())

            model.load_state_dict(best_state)
            score = test(
                model,
                test_loader,
                cfg["task"]
            )
            results[name][f"trial_{trial + 1}"] = score


            logger.info(f"trail:{trial + 1}: {name}: {score:.4f}")

        avg = np.mean(list(results[name].values()))
        std = np.std(list(results[name].values()))
        if cfg["metric"]=='rocauc':
            improvement = avg / molclr_baseline[name.lower()] - 1
        elif cfg['metric']=='rmse':
            improvement = molclr_baseline[name.lower()]/avg - 1
        all_improvement += improvement/len(my_task)
        logger.info(f"{name} - Average: {avg:.4f}, Std: {std:.4f}")
        results[name]["average"] = avg
        results[name]["std"] = std
        


    return results, all_improvement

if __name__ == "__main__":
    logits = torch.randn(4, 4)
    targets = torch.tensor([[1., 0., float('nan'), 1.], 
                            [0., float('nan'), 0., 1.],
                            [1., 1., 0., float('nan')],
                            [float('nan'), 0., 1., 0.]])

    loss_fn = MaskedBCEWithLogitsLoss()
    loss = loss_fn(logits, targets)
    print("Loss:", loss.item())