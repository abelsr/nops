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

from nops.fno.models.moe_fno import MoEFNO

torch.set_float32_matmul_precision('medium')

class VlasovPoissonDataModule(L.LightningDataModule):
    def __init__(
        self, 
        data_path: str, 
        batch_size: int = 8, 
        num_samples: int = 100, 
        workers: int = 8,
        pin_memory: bool = True,
        use_mmap: bool = True
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.workers = workers
        self.pin_memory = pin_memory
        self.use_mmap = use_mmap
        
    def setup(self, stage: Optional[str] = None):
        # Data shape: [samples, 64, 64, 64, 4]
        if self.data_path.endswith('.npy'):
            # Use memory mapping for large datasets
            mmap_mode = 'r' if self.use_mmap else None
            x = np.load(self.data_path, mmap_mode=mmap_mode)
            x = torch.tensor(x, dtype=torch.float32)
        elif self.data_path.endswith('.pt') or self.data_path.endswith('.pth'):
            x = torch.load(self.data_path)
        else:
            raise NotImplementedError("Unsupported file format. Use .npy or .pt/.pth files.")
        
        # Validate data
        assert torch.isfinite(x).all(), "Input contains NaN/Inf values"
        
        self.num_samples = x.shape[0]
        x_ = x[..., [0, 1, -1]]  # [samples, 64, 64, 64, 3]
        y  = x[..., [1]]  # [samples, 64, 64, 64, 1]
        
        # Permute to [samples, channels, *sizes] for consistency with FNO
        x_ = x_.permute(0, 4, 1, 2, 3)
        y = y.permute(0, 4, 1, 2, 3)
        
        # Validate shapes
        assert x_.shape[2:] == y.shape[2:], "Spatial dimensions must match"
        
        dataset = TensorDataset(x_, y)
        
        train_size = int(0.8 * self.num_samples)
        val_size = self.num_samples - train_size
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"\n{'='*60}")
        print(f"Dataset Configuration:")
        print(f"  Total samples: {self.num_samples}")
        print(f"  Train samples: {train_size}")
        print(f"  Val samples: {val_size}")
        print(f"  Input shape: {x_.shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Memory mapping: {self.use_mmap}")
        print(f"{'='*60}\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.workers > 0 else False
        )


class VisualizationCallback(L.Callback):
    def __init__(self, num_samples: int = 3, slice_idx: int = 32, log_every_n_epochs: int = 5):
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
            axes[i, 2].set_title(f"Absolute Error (Sample {i})")
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
        k: int = 2,
        routing_type: str = 'patch',
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
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
            k=k,
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


def main():
    # Set seeds for reproducibility
    L.seed_everything(42, workers=True, verbose=False) 
    
    # Data Module
    dm = VlasovPoissonDataModule(
        data_path="/home/ia/asantillan/Proyects/VlasovPoisson/dataset_pairs_float32.npy",
        batch_size=4,
        num_samples=10000,
        workers=2,
        pin_memory=True,
        use_mmap=True
    )
    
    # Model
    model = MoEFNOLitModule(
        modes=[32, 32, 32],
        num_moe_layers=4,
        num_experts=4,
        in_channels=3,
        out_channels=1,
        lifting_channels=64,
        projection_channels=64,
        mid_channels=32,
        expert_hidden_size=64,
        k=2,
        routing_type='patch',
        lr=5e-4,
        weight_decay=1e-4,
        warmup_epochs=50,
        max_epochs=1500
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/vlasov_moefno',
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
        slice_idx=32,
        log_every_n_epochs=5
    )
    
    # Logger
    logger = TensorBoardLogger(
        "lightning_logs", 
        name="vlasov_poisson_moefno",
        default_hp_metric=False
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=1500,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true',
        devices=[0, 1, 2],
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
            visualization_callback
        ],
        logger=logger,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision='16',  # or '16-mixed' for mixed precision
        deterministic=True,
        benchmark=True,  # Set to True if input sizes are fixed for speedup
    )
    
    # Print model summary
    print(f"\n{'='*60}")
    print(f"Model Configuration:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}\n")
    
    # Train
    trainer.fit(model, dm)
    
    # Load best model and test
    print(f"\nBest model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.6f}")


if __name__ == "__main__":
    main()