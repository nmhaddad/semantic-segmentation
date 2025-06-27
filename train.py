"""Trains a DeepLabv3 model from a configuration file"""

import os
from typing import Any, Dict

import torch
import yaml

import wandb
from models import DeepLabWrapper
from utils import Trainer, get_dataloaders

with open("config/config.yaml", "r") as f:
    config: Dict[str, Any] = yaml.safe_load(f)

# create an output directory for the model if one doesn't exist
os.makedirs("runs", exist_ok=True)

run = wandb.init(
    entity="nhaddad2112-duckasaurus",
    project="semantic-segmentation",
    config={
        "learning_rate": config.get("LEARNING_RATE", 1e-4),
        "batch_size": config.get("BATCH_SIZE", 16),
        "backbone": config.get("BACKBONE", "mobilenetv3large"),
        "dataset": "Yamaha",
        "epochs": config.get("NUM_EPOCHS", 25),
    },
)

dataloaders = get_dataloaders(config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
model = DeepLabWrapper(backbone=config["BACKBONE"], num_mask_channels=config["NUM_MASK_CHANNELS"])
class_weights = torch.tensor(config["CLASS_WEIGHTS"])
class_weights = class_weights.to("cuda")
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters, lr=float(config["LEARNING_RATE"]))
trainer = Trainer(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=config["NUM_EPOCHS"],
    logger=run,
    save_model_path=config.get("SAVE_MODEL_PATH"),
)
trainer.train()
