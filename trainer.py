import os
import random
import json

import numpy as np
import wandb
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torchvision

from dataloaders import get_dataloaders 
from losses import DiceBCELoss
from metrics import BatchSegmentationMetrics
from unet import *


class Trainer:
    def __init__(self, config_path):

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Seed
        seed = self.config["dataloader"].get("seed", 100)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # -----------------------------
        # Initialize model dynamically
        # -----------------------------
        self.model_name = self.config["model"].get("name", "Unet")
        if self.model_name not in globals():
            raise ValueError(f"Model {self.model_name} not found. Make sure it's imported or defined.")
        ModelClass = globals()[self.model_name]

        self.model = ModelClass(
            in_chans=self.config["model"].get("in_chans", 1),
            out_chans=self.config["model"].get("out_chans", 1),
            chans=self.config["model"].get("chans", 32),
            num_pool_layers=self.config["model"].get("num_pool_layers", 4),
            drop_prob=self.config["model"].get("drop_prob", 0.2),
            use_att=self.config["model"].get("use_att", False),
            use_res=self.config["model"].get("use_res", False)
        ).to(self.device)

        # -----------------------------
        # Dataloaders
        # -----------------------------
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=self.config["dataset"].get("data_dir"),
            splits_json=self.config["dataset"].get("splits_json"),
            fold=self.config["dataset"].get("fold", 0),
            modality=self.config["dataset"].get("modality", "both"),
            batch_size=self.config["dataloader"].get("batch_size", 8),
            num_workers=self.config["dataloader"].get("num_workers", 4),
            seed=seed
        )

        # -----------------------------
        # Loss
        # -----------------------------
        self.loss_name = self.config["training"].get("loss", "DiceBCELoss")

        if self.loss_name == "DiceBCELoss":
            self.criterion = DiceBCELoss().to(self.device)
        else:
            raise ValueError(f"Unknown loss {self.loss_name}")

        # -----------------------------
        # Optimizer
        # -----------------------------
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-5)
        )

        # -----------------------------
        # WandB
        # -----------------------------
        wandb.init(project=self.config["logging"].get("project", "MMI"),
                   config=self.config,
                   name=self.config["logging"].get("experiment_name", None))
        
        wandb.watch(self.model, log="all", log_freq=50)

        # Log model summary
        dummy_input = torch.randn(*self.config["training"].get("input_shape", [1, 1, 256, 256])).to(self.device)
        model_summary_str = str(summary(self.model, input_data=dummy_input))
        print(model_summary_str)
        wandb.log({"model_summary": wandb.Html(model_summary_str.replace("\n", "<br>"))})

        self.save_best_dir = self.config.get("ckpt_dir", "ckpts")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        loop = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.config.get('training', {}).get('epochs', 50)}]", leave=False)

        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)

            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

            loop.set_postfix({"loss": f"{loss.item():.4f}"})

            if batch_idx % 10 == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch})

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch, num_images=20):
        self.model.eval()
        val_loss = 0.0
        metrics = BatchSegmentationMetrics()
        
        all_preds = []
        all_targets = []

        # For image logging
        img_batches = []

        for batch_idx, (x, y) in enumerate(tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)):
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss = self.criterion(out, y)
            val_loss += loss.item()

            # Binarize predictions
            pred_bin = (torch.sigmoid(out) > 0.5).long()
            all_preds.append(pred_bin.cpu())
            all_targets.append(y.cpu())

            # Randomly save a batch for WandB logging
            if batch_idx in random.sample(range(len(self.val_loader)), k=min(num_images, len(self.val_loader))):
                # Take first few images in batch
                for i in range(min(x.size(0), num_images)):
                    img_batches.append(
                        wandb.Image(
                            x[i].cpu(),
                            masks={
                                "ground_truth": y[i].cpu(),
                                "prediction": pred_bin[i].cpu()
                            }
                        )
                    )

        val_loss /= len(self.val_loader)

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metric_dict = metrics.compute_all(all_preds, all_targets)

        log_dict = {"val_loss": val_loss, "epoch": epoch}
        log_dict.update(metric_dict)
        if img_batches:
            log_dict["val_images"] = img_batches
        wandb.log(log_dict)

        print(
            f"Epoch [{epoch+1}] Val Loss: {val_loss:.4f} | "
            f"Dice: {metric_dict['dice']:.4f} | "
            f"Precision: {metric_dict['precision']:.4f} | "
            f"Recall: {metric_dict['recall']:.4f} | "
            f"HD95: {metric_dict['hd95']:.4f}"
        )

        return val_loss

    def fit(self):
        epochs = self.config.get("epochs", 50)
        best_val = float("inf")

        for epoch in trange(epochs, desc="Training Epochs", unit="epoch"):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.save_best_dir, f"{self.model_name}_best_model.pth"))
                print("Saved best model!")