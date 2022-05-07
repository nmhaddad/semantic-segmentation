import os

import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import yaml

from utils import get_dataloader, Trainer


if __name__ == '__main__':

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # create an output directory for the model if one doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # load data into dataloaders
    dataloaders = get_dataloader(config['DATA_PATH'],
                                 batch_size=config['BATCH_SIZE'],
                                 resize_shape=(config['IMG_HEIGHT'], config['IMG_WIDTH']))

    # create the model
    if config['BACKBONE'] == 'RESNET101':
        model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                        progress=True,
                                                        aux_loss=True)
        model.classifier = DeepLabHead(2048, config['NUM_MASK_CHANNELS'])
    elif config['BACKBONE'] == 'RESNET50':
        model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                        progress=True,
                                                        aux_loss=True)
        model.classifier = DeepLabHead(2048, config['NUM_MASK_CHANNELS'])
    elif config['BACKBONE'] == 'MOBILENETV3LARGE':
        model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True,
                                                                progress=True,
                                                                aux_loss=True)
        model.classifier = DeepLabHead(960, config['NUM_MASK_CHANNELS'])
    else:
        print('Unknown backbone selected in configuration. Please select from RESNET50, RESNET101, or MOBILENETV3LARGE')
        raise ValueError

    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train the model
    trainer = Trainer(model, dataloaders, criterion, optimizer,
                      num_epochs=config['NUM_EPOCHS'],
                      is_inception=config['IS_INCEPTION'])
    trainer._train()

    # save the model
    model_path = config.get('SAVE_MODEL_PATH', 
                            'models/model_v1.{}.{}.pth'.format(
                                config['BATCH_SIZE'], 
                                config['NUM_EPOCHS']))
    torch.save(model, model_path)
