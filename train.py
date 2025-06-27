"""Trains a DeepLabv3 model from a configuration file"""

import os

import torch
import yaml

import wandb
from models import DeepLabWrapper
from utils import Trainer, get_dataloader

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# create an output directory for the model if one doesn't exist
os.makedirs("runs", exist_ok=True)

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="nhaddad2112-duckasaurus",
    # Set the wandb project where this run will be logged.
    project="semantic-segmentation",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": config.get("LEARNING_RATE", 1e-4),
        "batch_size": config.get("BATCH_SIZE", 16),
        "backbone": config.get("BACKBONE", "mobilenetv3large"),
        "dataset": "Yamaha",
        "epochs": config.get("NUM_EPOCHS", 25),
    },
)

# create dataloaders
dataloaders = get_dataloader(config["DATA_PATH"], batch_size=config["BATCH_SIZE"])


# create the model
model = DeepLabWrapper(backbone=config["BACKBONE"], num_mask_channels=config["NUM_MASK_CHANNELS"])

# train the model
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters, lr=float(config["LEARNING_RATE"]))
trainer = Trainer(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=config["NUM_EPOCHS"],
    logger=run,
)
trainer.train()

# save the model
model_path = config.get("SAVE_MODEL_PATH", f"models/{config['BACKBONE']}_v1.{config['NUM_EPOCHS']}.pth")
model.save_model(model_path)
