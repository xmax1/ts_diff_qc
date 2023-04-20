from typing import Optional, Any
from pydantic import BaseModel
import os
from cli import add_model
from rich.pretty import pprint
"""
```bash
docker container ls
docker container commit c16378f943fe new-image-name:latest

docker image ls
```

"""


class Pyfig(BaseModel):
    """Configuration for the training script."""

    data_path: str = "./data/"
    batch_size: int = 64
    lr: float = 1e-3
    n_epoch: int = 10
    n_worker: int = 4
    model_save_path: str = "./models/"
    wandb_project_name: str = "project-name"
    wandb_api_key: str = os.environ.get("WANDB_API_KEY")
    seed: int = 42
    accelerator: Optional[Any] = None
    checkpoint_frequency: int = 1
    
    log_model: bool = False

    data_split: list[float] = [0.8, 0.2]

sys_arg = add_model(Pyfig)
c = Pyfig(**vars(sys_arg))
pprint(c.dict())

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

    def __init__(self):
        super().__init__()
        self.lr = c.lr
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
        return optim.Adam(self.parameters(), lr=self.lr)



def main(c: Pyfig):
    """Main training function."""

    pl.seed_everything(c.random_seed)

    accelerator = Accelerator(
        device_placement=True,  # Automatically places tensors on the proper device
        fp16=True,  # Enables automatic mixed precision training (AMP)
        cpu=True,  # Forces the use of CPU even when GPUs are available
        split_batches=True,  # Splits the batches on the CPU before sending them to the device
        num_processes=1,  # Number of processes to use for distributed training (1 means no distributed training)
        local_rank=0,  # Local rank of the process (for distributed training)
    )

    dataset = CustomDataset(c.data_path)
    train_dataset, val_dataset = random_split(dataset, lengths= c.data_split)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        n_worker=c.n_worker,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=c.batch_size,
        n_worker=c.n_worker,
        shuffle=False,
        pin_memory=True
    )

    model = CustomModel(c.lr)

    wandb_logger = WandbLogger(
        project=c.wandb_project_name,
        log_model=c.log_model,
    )

    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=c.model_save_path,
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        save_last=True,
        period=c.checkpoint_frequency,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=c.n_epoch,
        accelerator=accelerator,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        gpus=torch.cuda.device_count(),
    )

    trainer.fit(model, train_loader, val_loader)

    wandb.finish()