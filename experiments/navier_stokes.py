import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from rich.table import Table
from rich.console import Console

from nops.fno.models.moe_fno import MoEFNO

torch.set_float32_matmul_precision('medium')

class VlasovPoissonDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 16, num_workers: int = 4, pin_memory: bool = True):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage: Optional[str] = None):
        if self.data_path.endswith('.npz'):
            data = np.load(self.data_path)
            u = data['u']  # Shape: (num_samples, 64, 64, 10) [samples, x, y, t]
        elif self.data_path.endswith('.npy'):
            u = np.load(self.data_path)  # Shape: (num_samples, 64, 64, 10)
        elif self.data_path.endswith('.pt'):
            data = torch.load(self.data_path)
            u = data['u'].numpy()  # Shape: (num_samples, 64, 64, 10)
        else:
            raise NotImplementedError("Unsupported file format. Use .npz, .npy, or .pt files.")
        
        num_samples = u.shape[0]
        train_size = int(0.8 * num_samples)
        val_size = num_samples - train_size
        
        x = u[:, :, :, :10]  # Input: first 5 time steps
        y = u[:, :, :, 10:]  # Output: last 5 time steps
        
        # Add channel dim
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_samples, 64, 64, 5, 1)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # Shape: (num_samples, 64, 64, 5, 1)
        
        # Permute to [samples, channels, *sizes] for consistency with FNO
        x = x.permute(0, 4, 1, 2, 3)  # Add channel of 'channels0' [samples, 1, 64, 64, 5]
        y = y.permute(0, 4, 1, 2, 3)  # Add channel of 'channels0' [samples, 1, 64, 64, 5]
        
        dataset = TensorDataset(x, y)
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # print summary as a table
        table = Table(title="Dataset Summary")
        table.add_column("Split", justify="center", style="cyan", no_wrap=True)
        table.add_column("Number of Samples", justify="center", style="magenta")
        table.add_row("Train", str(len(self.train_dataset)))
        table.add_row("Validation", str(len(self.val_dataset)))
        console = Console()
        console.print(table)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory
        )
        
        
class VisualizationCallback(L.Callback):
    def __init__(self, num_samples: int = 3, slice_idx: int = 5, log_every_n_epochs: int = 5):
        super().__init__()
        self.num_samples = num_samples
        self.slice_idx = slice_idx
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip during sanity check
        if trainer.sanity_checking:
            return
        
        # Only log every N epochs to save time/space
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Get a batch from validation set
        val_loader = trainer.val_dataloaders
        batch = next(iter(val_loader)) # type: ignore
        
        x, y = batch
        x = x.to(pl_module.device)
        y = y.to(pl_module.device)
        
        pl_module.eval()
        with torch.no_grad():
            y_hat = pl_module(x)
        
        # Move to CPU and numpy
        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        
        num_to_plot = min(self.num_samples, y.shape[0])
        fig, axes = plt.subplots(num_to_plot, 3, figsize=(15, 5 * num_to_plot))
        
        # Handle case where num_to_plot is 1
        if num_to_plot == 1:
            axes = axes[np.newaxis, :]

        for i in range(num_to_plot):
            # Take a 2D slice (middle of the 3rd spatial dimension)
            gt_slice = y[i, 0, :, :, self.slice_idx]
            pred_slice = y_hat[i, 0, :, :, self.slice_idx]
            error_slice = np.abs(gt_slice - pred_slice)
            err_min, err_max = error_slice.min(), error_slice.max()
            if err_max > err_min:
                error_slice = (error_slice - err_min) / (err_max - err_min)
            else:
                error_slice = np.zeros_like(error_slice)
            
            vmin, vmax = gt_slice.min(), gt_slice.max()
            
            im0 = axes[i, 0].imshow(gt_slice, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f"Ground Truth (Sample {i})")
            axes[i, 0].axis('off')
            fig.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            
            im1 = axes[i, 1].imshow(pred_slice, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f"Prediction (Sample {i})")
            axes[i, 1].axis('off')
            fig.colorbar(im1, ax=axes[i, 1], fraction=0.046)
            
            im2 = axes[i, 2].imshow(error_slice, cmap='magma')
            axes[i, 2].set_title(f"Relative Error (Sample {i})")
            axes[i, 2].axis('off')
            fig.colorbar(im2, ax=axes[i, 2], fraction=0.046)
        
        plt.tight_layout()
        
        # Log to logger if available
        if trainer.logger:
            if hasattr(trainer.logger.experiment, 'add_figure'):  # TensorBoard # type: ignore
                trainer.logger.experiment.add_figure( # type: ignore
                    'Validation/Predictions', 
                    fig, 
                    global_step=trainer.global_step
                )
        
        plt.close(fig)
        
class MoEFNOLitModule(L.LightningModule):
    def __init__(
        self,
        modes: List[int] = [16, 16, 16],
        num_moe_layers: int = 4,
        num_experts: int = 4,
        in_channels: int = 4,
        lifting_channels: int = 64,
        projection_channels: int = 64,
        out_channels: int = 1,
        mid_channels: int = 32,
        expert_hidden_size: int = 64,
        top_k: int = 2,
        routing_type: str = 'patch',
        lr: float = 5e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = MoEFNO(
            modes=modes,
            num_moe_layers=num_moe_layers,
            num_experts=num_experts,
            in_channels=in_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            expert_hidden_size=expert_hidden_size,
            top_k=top_k,
            routing_type=routing_type,
            add_grid=True
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # Track best validation loss
        self.best_val_loss = float('inf')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Main loss
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        
        # Combined loss
        loss = mse_loss + 0.1 * mae_loss
        
        # Relative error
        rel_error = torch.mean(torch.abs(y_hat - y) / (torch.abs(y) + 1e-8))
        
        # Logging
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log("train_mse", mse_loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log("train_mae", mae_loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log("train_rel_error", rel_error, sync_dist=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Compute multiple metrics
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        rel_error = torch.mean(torch.abs(y_hat - y) / (torch.abs(y) + 1e-8))
        
        # Max absolute error
        max_error = torch.max(torch.abs(y_hat - y))
        
        # Logging
        self.log("val_loss", mse_loss, prog_bar=True, sync_dist=True)
        self.log("val_mae", mae_loss, sync_dist=True)
        self.log("val_rel_error", rel_error, sync_dist=True)
        self.log("val_max_error", max_error, sync_dist=True)
        
        return mse_loss
    
    def on_validation_epoch_end(self):
        # Track best validation loss
        val_loss = self.trainer.callback_metrics.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.log("best_val_loss", self.best_val_loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Warmup + Cosine Annealing
        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
        
        
if __name__ == "__main__":
    # Set seeds for reproducibility
    L.seed_everything(42, workers=True, verbose=False) 
    
    # DataModule
    data_module = VlasovPoissonDataModule(
        data_path="/home/abelsr/Proyects/Deep-Learning/nops/navier_stokes_20_t.pt",
        batch_size=16,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = MoEFNOLitModule(
        modes=[16, 16, 8],
        num_moe_layers=4,
        num_experts=3,
        in_channels=1,
        out_channels=1,
        lifting_channels=128,
        projection_channels=128,
        mid_channels=64,
        expert_hidden_size=32,
        top_k=2,
        routing_type='patch',
        lr=1e-3,
        weight_decay=1e-4,
        warmup_epochs=50,
        max_epochs=500
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/navier_stokes_moefno',
        filename='moefno-{epoch:03d}-{val_loss:.6f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
        verbose=False
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=100,
        mode='min',
        verbose=False
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    visualization_callback = VisualizationCallback(
        num_samples=3, 
        slice_idx=-1,
        log_every_n_epochs=5
    )
    
    # Logger
    logger = TensorBoardLogger(
        "lightning_logs", 
        name="navier_stokes_moefno",
        default_hp_metric=False
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=500,
        accelerator="gpu",
        devices=1,
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            visualization_callback
        ],
        logger=logger,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision='16-mixed',  # or '16-mixed' for mixed precision
        # deterministic=True,
        benchmark=True,  # Set to True if input sizes are fixed for speedup
    )
    
    console = Console()

    # Model summary table
    model_table = Table(title="Model Configuration", show_header=False)
    model_table.add_column("Metric", style="cyan")
    model_table.add_column("Value", style="magenta", justify="right")
    model_table.add_row("Total parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    model_table.add_row("Trainable parameters", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    console.print(model_table)
    
    # Train
    trainer.fit(model, data_module)
    
    # Training results table
    res_table = Table(title="Training Summary", show_header=False)
    res_table.add_column("Metric", style="cyan")
    res_table.add_column("Value", style="green")
    res_table.add_row("Best model checkpoint", str(checkpoint_callback.best_model_path))
    res_table.add_row("Best validation loss", f"{checkpoint_callback.best_model_score:.6f}")
    console.print(res_table)
