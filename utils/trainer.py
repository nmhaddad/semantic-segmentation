"""Class for training DeepLab models"""

import copy
import time

import torch
from tqdm import tqdm

from models import DeepLabWrapper


class Trainer:
    """This class trains DeepLab models given a configuration of hyperparameters

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

    """

    def __init__(
        self,
        deeplab: DeepLabWrapper,
        dataloaders: torch.utils.data.DataLoader,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        num_epochs: int = 25,
        logger=None,
    ):
        """Initialization method for Trainer base class

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

        """
        self.deeplab = deeplab
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.logger = logger

    def train(self) -> None:
        """This function is used to train a model

        Returns:
            model, val_mean_iou_history
        """
        since = time.time()
        from torchmetrics.segmentation import MeanIoU

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        val_mean_iou_history = []
        best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())
        best_mean_iou = 0.0
        self.deeplab.model.to(device)
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 10)

            for phase in ["train", "valid"]:
                if phase == "train":
                    self.deeplab.model.train()
                else:
                    self.deeplab.model.eval()

                mean_iou = MeanIoU(num_classes=self.deeplab.num_mask_channels).to(device)
                running_loss = 0.0

                # Iterate over data.
                for inputs, labels in tqdm(iter(self.dataloaders[phase])):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        # Get model outputs and calculate loss
                        outputs = self.deeplab.model(inputs)
                        loss = self.criterion(outputs["out"], labels)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    mean_iou.update(torch.argmax(outputs["out"], 1), labels)
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_mean_iou = mean_iou.compute().item()

                if self.logger:
                    self.logger.log(
                        {
                            f"{phase}_loss": epoch_loss,
                            f"{phase}_mean_iou": epoch_mean_iou,
                            "epoch": epoch + 1,
                        }
                    )

                print(f"{phase} Loss: {epoch_loss:.4f} mIoU: {epoch_mean_iou:.4f}")
                # deep copy the model
                if phase == "valid" and epoch_mean_iou > best_mean_iou:
                    best_mean_iou = epoch_mean_iou
                    best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())
                if phase == "valid":
                    val_mean_iou_history.append(epoch_mean_iou)

            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val mean IoU: {best_mean_iou:4f}")

        # load best model weights
        self.deeplab.model.load_state_dict(best_model_wts)

        if self.logger:
            self.logger.finish()

        return self.deeplab, val_mean_iou_history
