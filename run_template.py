from pydantic import BaseModel
from typing import Optional, Any
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchmetrics
import pytorch_lightning as pl
from accelerate import Accelerator
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

class Config(BaseModel):
    """Configuration for the training script."""

    data_path: str = "./data/"
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    num_workers: int = 4
    model_save_path: str = "./models/"
    wandb_project_name: str = "project-name"
    wandb_api_key: str = "your_wandb_api_key"
    random_seed: int = 42
    accelerator: Optional[Any] = None
    checkpoint_frequency: int = 1

    log_model: bool = False

class CustomDataset(Dataset):
    """Custom dataset class."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        # Load your data from data_path

    def __len__(self) -> int:
        # Return the length of the dataset
        pass

    def __getitem__(self, idx: int) -> tuple:
        # Return a single data sample
        pass

class CustomModel(pl.LightningModule):
    """Custom model class."""

    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = nn.Sequential(
            # Add your model layers here
        )
        self.loss_function = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)

c = Config()

def main(cfg: Config):
    """Main training function."""

    pl.seed_everything(cfg.random_seed)

    accelerator = Accelerator(
        device_placement=True,  # Automatically places tensors on the proper device
        fp16=True,  # Enables automatic mixed precision training (AMP)
        cpu=True,  # Forces the use of CPU even when GPUs are available
        split_batches=True,  # Splits the batches on the CPU before sending them to the device
        num_processes=1,  # Number of processes to use for distributed training (1 means no distributed training)
        local_rank=0,  # Local rank of the process (for distributed training)
    )
    cfg.accelerator = accelerator

    dataset = CustomDataset(cfg.data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True
    )

    model = CustomModel(cfg.learning_rate)

    wandb_logger = WandbLogger(
        project=cfg.wandb_project_name,
        log_model=c.log_model,
    )

    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.model_save_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        save_last=True,
        period=cfg.checkpoint_frequency,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator=cfg.accelerator,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gpus=torch.cuda.device_count(),
    )

    trainer.fit(model, train_loader, val_loader)

    wandb.finish()
