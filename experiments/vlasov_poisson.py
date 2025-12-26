import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
from nops.fno.models.moe_fno import MoEFNO
from typing import List, Any, Optional

torch.set_float32_matmul_precision('medium')

class VlasovPoissonDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 8, num_samples: int = 100, workers: int = 8):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.workers = workers
        
    def setup(self, stage: Optional[str] = None):
        # Data shape: [samples, 64, 64, 64, 4]
        if self.data_path.endswith('.npy'):
            x = np.load(self.data_path)
            x = torch.tensor(x, dtype=torch.float32)
        elif self.data_path.endswith('.pt') or self.data_path.endswith('.pth'):
            x = torch.load(self.data_path)
        else:
            raise NotImplementedError("Unsupported file format. Use .npy or .pt/.pth files.")
        
        self.num_samples = x.shape[0]
        x_ = x[..., [0, 1, -1]]  # [samples, 64, 64, 64, 3]
        y  = x[..., [1]]  # [samples, 64, 64, 64, 1]
        
        # Permute to [samples, channels, *sizes] for consistency with FNO
        x_ = x_.permute(0, 4, 1, 2, 3)
        y = y.permute(0, 4, 1, 2, 3)
        
        dataset = TensorDataset(x_, y)
        
        train_size = int(0.8 * self.num_samples)
        val_size = self.num_samples - train_size
        self.train_ds, self.val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.workers)

class VisualizationCallback(L.Callback):
    def __init__(self, num_samples: int = 3, slice_idx: int = 32):
        super().__init__()
        self.num_samples = num_samples
        self.slice_idx = slice_idx

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Only plot for the first batch of validation
        if batch_idx == 0:
            x, y = batch
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
                fig.colorbar(im0, ax=axes[i, 0])
                
                im1 = axes[i, 1].imshow(pred_slice, cmap='viridis', vmin=vmin, vmax=vmax)
                axes[i, 1].set_title(f"Prediction (Sample {i})")
                fig.colorbar(im1, ax=axes[i, 1])
                
                im2 = axes[i, 2].imshow(error_slice, cmap='magma')
                axes[i, 2].set_title(f"Absolute Error (Sample {i})")
                fig.colorbar(im2, ax=axes[i, 2])
            
            plt.tight_layout()
            
            # Log to logger if available
            if trainer.logger:
                if hasattr(trainer.logger.experiment, 'add_figure'): # TensorBoard
                    trainer.logger.experiment.add_figure('Validation/Predictions', fig, global_step=trainer.global_step)
                elif hasattr(trainer.logger, 'log_image'): # WandB
                    trainer.logger.log_image(key='Validation/Predictions', images=[fig])
            
            # Also save locally
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
        lr: float = 1e-3
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():
    # Hyperparameters
    L.seed_everything(42)
    
    dm = VlasovPoissonDataModule(
        data_path="/home/ia/asantillan/Proyects/VlasovPoisson/dataset_pairs_float32.npy",
        batch_size=4,
        num_samples=10000,
        workers=2
    )
    
    model = MoEFNOLitModule(
        modes=[16, 16, 16],
        num_moe_layers=4,
        num_experts=8,
        in_channels=3,
        out_channels=1,
        mid_channels=32,
        k=2,
        routing_type='patch',
        lr=1e-4
    )
    
    trainer = L.Trainer(
        max_epochs=500,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true',
        devices=[0,1,2],
        callbacks=[VisualizationCallback(num_samples=3, slice_idx=32)],
        logger=[TensorBoardLogger("lightning_logs", name="vlasov_poisson_moefno")],
        accumulate_grad_batches=2,
        # precision="16-mixed"
    )
    
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
