import os
from pydantic import BaseModel, Field, validator, Extra, fields, BaseSettings
import numpy as np
from rich import inspect
from rich.pretty import pprint
from pathlib import Path
from pyfig.dashboard import run_dash
from pyfig.cli import add_model
from pyfig.files import lo_ve
from pyfig.schema import safe_pydanmod_to_schema
from pyfig.utils import mega_inspect_pydanmod, load_env_vars
from pyfig import Pyfig

from pyfig.plugins import Logger, Paths, d_to_wb, Env
from pyfig.typfig import ModdedModel, ndarray

from hwat import Logger, System, Ansatz, Walkers, Opt, MULTIMODE

class Pyfig(ModdedModel):
	
	env: Env 				= Env(_env_file= '.env')
	seed: int           	= 42
	device          		= 'cpu'
	dtype: str          	= 'float32'
	
	pth: Paths = Paths(
		project         = 'hwat',
		exp_name        = 'test',
	)

	mode: MULTIMODE       	= MULTIMODE.train  # adam, sgd
	n_step: int      		= 100

	loss: str        		= 'vmc'  # orb_mse, vmc
	compute_energy: bool 	= False  # true by default

	dashboard: bool       = False

	sys: System 		= System(
			a = np.array([[1.0, 1.0, 1.0]]),
			a_z = np.array([1]),
	)

	anz: Ansatz      	= Ansatz(
		n_l= 2,
	)
	wlk: Walkers     	= Walkers(
		# n_b= 128,
	)
	wnb: Logger      	= Logger(
		exp_name = pth.exp_name, 
		entity= env.entity,
		project= pth.project,
		n_log_metric= 10,
		n_log_state= -1,
		run_path= pth.run_path,
		exp_data_dir= pth.exp_data_dir,
	)
	opt: Opt         	= Opt(
		name= 'AdamW',
	)


pyfig = Pyfig()
# sch = pyfig.sys.schema_json()
# pprint(sch)
# # pyfig.dict()
# # inspect(pyfig, all= True)

# exit()

# mega_inspect_pydanmod(pyfig, 'HWAT', show= True, break_on_error= True, mega= True) # fails on json schema, I think because nested
# args = add_model(pyfig)
# params = vars(args)

# exit()
# run
from accelerate import Accelerator
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics

from runner import Ltng


pyfig = Pyfig()
pprint(pyfig.dict())
exit()

pl.seed_everything(pyfig.seed)

wandb_logger = WandbLogger(
	project=pyfig.pth.project,
	entity=pyfig.env.entity,
	save_dir= pyfig.pth.exp_data_dir,
)

lightning = Ltng(
	pyfig= 	pyfig,
)

accelerator = Accelerator(
	device_placement=True,  # Automatically places tensors on the proper device
	fp16=True,  # Enables automatic mixed precision training (AMP)
	cpu=True,  # Forces the use of CPU even when GPUs are available
	split_batches=True,  # Splits the batches on the CPU before sending them to the device
	num_processes=1,  # Number of processes to use for distributed training (1 means no distributed training)
	local_rank=0,  # Local rank of the process (for distributed training)
)

from hwat import Ansatz_fb as Model
from hwat import sample_b

model = Model(pyfig.lr)

wandb_logger.watch(model)

trainer = pl.Trainer(
	accelerator= accelerator,
	logger= wandb_logger,
	gpus=torch.cuda.device_count(),
	# callbacks= [checkpoint_callback],
)

trainer.fit(model) # train_loader

wandb.finish()