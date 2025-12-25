import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from nops.fno.models.moe_fno import MoEFNO
from typing import List, Any, Optional

class VlasovPoissonDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 8, num_samples: int = 100):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_samples = num_samples

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
        x_ = x[..., :3]
        y  = x[..., 3:]
        
        dataset = TensorDataset(x_, y)
        
        train_size = int(0.8 * self.num_samples)
        val_size = self.num_samples - train_size
        print(f"Train size: {train_size}, Val size: {val_size}")
        self.train_ds, self.val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=4)

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
        x = x.permute(0, -1, *range(1, x.dim() -1))  # [batch, channels, *sizes]
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

def main():
    # Hyperparameters
    L.seed_everything(42)
    
    dm = VlasovPoissonDataModule(
        data_path="/home/ia/asantillan/Proyects/VlasovPoisson/dataset_pairs_float32.npy",
        batch_size=4,
        num_samples=10000
    )
    
    model = MoEFNOLitModule(
        modes=[8, 8, 8],
        num_moe_layers=2,
        num_experts=4,
        in_channels=3,
        out_channels=1,
        mid_channels=32,
        k=1,
        routing_type='patch',
        lr=5e-5
    )
    
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=-1,
        # precision="16-mixed"
    )
    
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
