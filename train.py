""" Trains a DeepLabv3 model from a configuration file """

import os

import torch
import yaml

from models import DeepLabWrapper
from utils import get_dataloader, Trainer


if __name__ == '__main__':

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # create an output directory for the model if one doesn't exist
    os.makedirs('runs', exist_ok=True)

    # create dataloaders
    dataloaders = get_dataloader(config['DATA_PATH'],
                                 batch_size=config['BATCH_SIZE'],
                                 resize_shape=(config['IMG_HEIGHT'], config['IMG_WIDTH']))

    # create the model
    model = DeepLabWrapper(backbone=config['BACKBONE'], num_mask_channels=config['NUM_MASK_CHANNELS'])

    # train the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters, lr=1e-4)
    trainer = Trainer(model, dataloaders, criterion, optimizer,
                      num_epochs=config['NUM_EPOCHS'],
                      is_inception=config['IS_INCEPTION'])
    trainer.train()

    # save the model
    model_path = config.get('SAVE_MODEL_PATH', f'models/{config["BACKBONE"]}_v1.{config["NUM_EPOCHS"]}.pth')
    model.save_model(model_path)
