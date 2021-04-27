import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import os
import time
import copy
from tqdm import tqdm

from utils import mean_iou
from dataset import get_dataloader

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    """ This function is used to train a model

    Args:
        model: (torchvision.models.segmentation.deeplabv3) the model used in training
        criterion: (torch.nn.CrossEntropyLoss) the loss function to use
        dataloaders: (torch.utils.data.Dataloader) the dataloader to use
        optimizer: (torch.optim.Adam) the optimizer to use
        num_epochs: (int=25) the number of epochs to train
        is_inception: (bool) whether or not to use auxiliary outputs in training

    Returns:
        model, val_mean_iou_history
    """
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_mean_iou_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_mean_iou = 0.0
    model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_mean_iou = 0

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):

                inputs = sample['image'].to(device)
                labels = sample['mask'].to(device)

                label = torch.argmax(labels, dim=1)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if is_inception and phase == 'train':
                        outputs = model(inputs)
                        loss1 = criterion(outputs['out'], label)
                        loss2 = criterion(outputs['aux'], label)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        outputs['out'] = outputs['out'].to(device)
                        loss = criterion(outputs['out'], label)
                    _, preds = torch.max(outputs['out'], 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_mean_iou += mean_iou(torch.argmax(outputs['out'], 1), label).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_mean_iou = running_mean_iou / len(dataloaders[phase])
            print('{} Loss: {:.4f} mIoU: {:.4f}'.format(phase, epoch_loss, epoch_mean_iou))
            # deep copy the model
            if phase == 'valid' and epoch_mean_iou > best_mean_iou:
                best_mean_iou = epoch_mean_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'valid':
                val_mean_iou_history.append(epoch_mean_iou)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val mean IoU: {:4f}'.format(best_mean_iou))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_mean_iou_history


if __name__ == '__main__':

    # path variables
    DATA_PATH = 'data/yamaha_v0/'
    MODEL_PATH = 'models/model_v0.1.pt'

    # image resize variables
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    epochs = 1
    num_mask_channels = 8
    batch_size = 2

    # create an output directory for the model if one doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # load data into dataloaders
    dataloaders = get_dataloader(DATA_PATH,
                                 batch_size=batch_size,
                                 resize_shape=(IMG_HEIGHT, IMG_WIDTH))

    # create the model (3 types are shown below. Uncomment the one you want use)
    # model = models.segmentation.deeplabv3_resnet101(pretrained=True,
    #                                                 progress=True,
    #                                                 aux_loss=True)
    # model.classifier = DeepLabHead(2048, num_mask_channels)

    model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                    progress=True,
                                                    aux_loss=True)
    model.classifier = DeepLabHead(2048, num_mask_channels)

    # model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True,
    #                                                          progress=True,
    #                                                          aux_loss=True)
    # model.classifier = DeepLabHead(960, num_mask_channels)

    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train the model
    train_model(model, dataloaders, criterion, optimizer, num_epochs=epochs, is_inception=True)

    # save the model
    torch.save(model, MODEL_PATH)
