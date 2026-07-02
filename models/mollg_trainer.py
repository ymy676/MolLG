import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from tqdm import tqdm
from utils.checkpoint import save_checkpoint
from utils.metrics import binary_classification_metrics, multitask_binary_classification_metrics, regression_metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from utils.finetuner import get_loaders

class mollg_trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        logger,
        writer,
        exp_dir,
        cfg
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        self.writer = writer
        self.exp_dir = exp_dir
        self.cfg = cfg
        self.history = []
        self.gamma = cfg.model.gamma
        self.best_rmse = float("inf")
        self.best_probe_score = float("-inf")

    def train_epoch(self,loader,epoch=None):
        self.model.train()
        train_loss = 0.0
        recon_loss = 0.0
        clr_loss = 0.0

        for graph, xis, xjs in tqdm(loader, desc="Training"):
            graph, xis, xjs = graph.to(self.device), xis.to(self.device), xjs.to(self.device)
            loss, loss_dict, gamma = self.model(graph, xis, xjs, gamma=self.gamma, epoch=epoch)

            train_loss += loss.item()
            recon_loss += loss_dict['mae_loss']
            clr_loss += loss_dict['clr_loss']



            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        train_loss /= len(loader)
        recon_loss /= len(loader)
        clr_loss /= len(loader)  

        return train_loss, recon_loss, clr_loss, gamma
    
    @torch.no_grad()
    def eval_epoch(self, loader, epoch=None):
        self.model.eval()
        val_loss = 0.0
        val_recon_loss = 0.0
        val_clr_loss = 0.0

        for graph, xis, xjs in tqdm(loader, desc="Evaluating"):
            graph, xis, xjs = graph.to(self.device), xis.to(self.device), xjs.to(self.device)
            loss, loss_dict, gamma = self.model(graph, xis, xjs, gamma=self.gamma, epoch=epoch)

            val_loss += loss.item()
            val_recon_loss += loss_dict['mae_loss']
            val_clr_loss += loss_dict['clr_loss']

        
        val_loss /= len(loader)
        val_recon_loss /= len(loader)
        val_clr_loss /= len(loader)  

        return val_loss, val_recon_loss, val_clr_loss, gamma

    def save_metrics(self, epoch, train_loss, recon_loss, clr_loss, val_loss, val_recon_loss, val_clr_loss, metrics=None):
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "recon_loss": recon_loss,
            "clr_loss": clr_loss,
            "val_loss": val_loss,
            "val_recon_loss": val_recon_loss,
            "val_clr_loss": val_clr_loss,
        }

        if metrics is not None:
            row.update({
                "rmse": metrics["RMSE"],
                "mae": metrics["MAE"],
                "r2": metrics["R2"]
            })

        self.history.append(row)

        pd.DataFrame(self.history).to_csv(
            f"{self.exp_dir}/metrics.csv",
            index=False
        )
    
    def extract_representations(self, loader):
        self.model.eval()
        representations = []
        targets = []

        for graph in tqdm(loader, desc="Extracting Representations"):
            graph = graph.to(self.device)
            with torch.no_grad():
                rep, _, _ = self.model.encoder(graph)
            representations.append(rep.cpu())
            targets.append(graph.y.cpu())

        representations = torch.cat(representations, dim=0)
        targets = torch.cat(targets, dim=0)

        return representations.numpy(), targets.numpy()
    
    def linear_probe(self, train_loader, val_loader, task_type="Regression"):
        # Extract representations
        train_reps, train_targets = self.extract_representations(train_loader)
        val_reps, val_targets = self.extract_representations(val_loader)
        train_targets = train_targets.squeeze()
        val_targets = val_targets.squeeze()

        # Scale the features
        scaler = StandardScaler()
        train_reps = scaler.fit_transform(train_reps)
        val_reps = scaler.transform(val_reps)

        # Train linear regression
        if task_type == "Regression":
            reg = LinearRegression().fit(train_reps, train_targets)
            val_preds = reg.predict(val_reps)
            metrics = regression_metrics(val_targets, val_preds)
        elif task_type == "BinaryClassification":
            reg = LogisticRegression(max_iter=1000).fit(train_reps, train_targets)
            val_preds = reg.predict_proba(val_reps)[:, 1]
            metrics = binary_classification_metrics(val_targets, val_preds)
        elif task_type == "MultiOutputBinaryClassification":
            reg = MultiOutputClassifier(LogisticRegression(max_iter=1000)).fit(train_reps, train_targets)
            val_preds = reg.predict_proba(val_reps)
            metrics = multitask_binary_classification_metrics(val_targets, val_preds)


        return metrics

    def pretrain(self, train_loader, val_loader, epochs):

        
        
        for epoch in range(1, epochs + 1):
            train_loss, recon_loss, clr_loss, gamma = self.train_epoch(train_loader, epoch=epoch)
            val_loss, val_recon_loss, val_clr_loss, gamma = self.eval_epoch(val_loader, epoch=epoch)

            self.logger.info(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss:.4f} (Recon: {recon_loss:.4f}, CLR: {clr_loss:.4f}) - "
                f"Val Loss: {val_loss:.4f} (Recon: {val_recon_loss:.4f}, CLR: {val_clr_loss:.4f})"
            )

            self.writer.add_scalars("loss/total", {"train": train_loss, "val": val_loss,}, epoch)
            self.writer.add_scalars("loss/recon", {"train": recon_loss, "val": val_recon_loss,}, epoch)
            self.writer.add_scalars("loss/clr", {"train": clr_loss, "val": val_clr_loss,}, epoch)
            self.writer.add_scalar("gamma", gamma, epoch)
            # Save checkpoint
            if epoch % self.cfg.train.probe_freq == 0:

                probe_results = {}

                for name, task_type in {
                    'bbbp': "BinaryClassification",
                    'bace': "BinaryClassification",
                    'clintox': "MultiOutputBinaryClassification",
                    'freesolv': "Regression",
                    'esol': "Regression",
                    'lipo': "Regression"
                }.items():

                    self.logger.info(f"Evaluating on {name}...")

                    probe_train_loader, probe_val_loader, _ = get_loaders(
                        root="data/finetune",
                        name=name
                    )

                    metrics = self.linear_probe(
                        probe_train_loader,
                        probe_val_loader,
                        task_type=task_type
                    )

                    # -----------------------
                    # store raw metrics
                    # -----------------------
                    if task_type == "Regression":
                        score = -metrics["RMSE"]   # unify direction
                        self.logger.info(f"{name} RMSE: {metrics['RMSE']:.4f}")
                    else:
                        score = metrics["ROC-AUC"]
                        self.logger.info(f"{name} ROC-AUC: {metrics['ROC-AUC']:.4f}")

                    probe_results[name] = {
                        "score": score,
                        "metrics": metrics,
                        "task_type": task_type
                    }

                # =====================================================
                # 1. compute unified score (for checkpoint selection)
                # =====================================================
                scores = [v["score"] for v in probe_results.values()]
                unified_score = sum(scores) / len(scores)

                self.logger.info(f"Unified Linear Probe Score: {unified_score:.4f}")
                self.writer.add_scalar("probe/unified_score", unified_score, epoch)

                # =====================================================
                # 2. checkpoint selection
                # =====================================================
                if unified_score > self.best_probe_score:
                    self.best_probe_score = unified_score

                    self.logger.info(f"New Best Probe Score: {unified_score:.4f}")

                    save_checkpoint(
                        path=os.path.join(self.exp_dir, "best_probe_checkpoint.pth"),
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                    )
        save_checkpoint(
            path=os.path.join(self.exp_dir, "last_checkpoint.pth"),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
        )

     


