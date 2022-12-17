""" Class for training DeepLab models """

import time
import copy

import torch
from tqdm import tqdm

from .utils import mean_iou
from models import DeepLabWrapper


class Trainer:
    """ This class trains DeepLab models given a configuration of hyperparameters 

    Attributes:
        deeplab: DeepLabWrapper
            Model to train
        dataloaders: torch.utils.data.DataLoader
            Dataloaders to use for training
        criterian: torch.nn.CrossEntropyLoss
            Loss function to use
        optimizer: torch.optim.Adam
            Optimizer to use
        num_epochs: int
            Number of epochs to train
        is_inception: bool
            Use auxiliary outputs and loss during training
        """

    def __init__(self, deeplab: DeepLabWrapper, dataloaders: torch.utils.data.DataLoader,
                 criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Adam, num_epochs: int = 25,
                 is_inception: bool = False):
        """ Initialization method for Trainer base class

        Args:
            model: (torchvision.models.segmentation.deeplabv3)
                the model used in training
            dataloaders: (torch.utils.data.DataLoader)
                the dataloader to use
            criterion: (torch.nn.CrossEntropyLoss)
                the loss function to use
            optimizer: (torch.optim.Adam)
                the optimizer to use
            num_epochs: (int=25)
                the number of epochs to train
            is_inception: (bool)
                whether or not to use auxiliary outputs in training
        """
        self.deeplab = deeplab
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.is_inception = is_inception

    def train(self) -> None:
        """ This function is used to train a model

        Returns:
            model, val_mean_iou_history
        """
        self.deeplab.model.train()
        since = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        val_mean_iou_history = []
        best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())
        best_mean_iou = 0.0
        self.deeplab.model.to(device)
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/{self.num_epochs}')
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.deeplab.model.train()
                else:
                    self.deeplab.model.eval()

                running_loss = 0.0
                running_mean_iou = 0

                # Iterate over data.
                for sample in tqdm(iter(self.dataloaders[phase])):

                    inputs = sample['image'].to(device)
                    labels = sample['mask'].to(device)

                    label = torch.argmax(labels, dim=1)

                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        if self.is_inception and phase == 'train':
                            outputs = self.deeplab.model(inputs)
                            loss1 = self.criterion(outputs['out'], label)
                            loss2 = self.criterion(outputs['aux'], label)
                            loss = loss1 + 0.4 * loss2
                        else:
                            outputs = self.deeplab.model(inputs)
                            outputs['out'] = outputs['out'].to(device)
                            loss = self.criterion(outputs['out'], label)
                        _, preds = torch.max(outputs['out'], 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_mean_iou += mean_iou(torch.argmax(outputs['out'], 1), label).item()

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_mean_iou = running_mean_iou / len(self.dataloaders[phase])
                print('{} Loss: {:.4f} mIoU: {:.4f}'.format(phase, epoch_loss, epoch_mean_iou))
                # deep copy the model
                if phase == 'valid' and epoch_mean_iou > best_mean_iou:
                    best_mean_iou = epoch_mean_iou
                    best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())
                if phase == 'valid':
                    val_mean_iou_history.append(epoch_mean_iou)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val mean IoU: {:4f}'.format(best_mean_iou))

        # load best model weights
        self.deeplab.model.load_state_dict(best_model_wts)
        return self.deeplab, val_mean_iou_history
