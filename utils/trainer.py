"""Class for training DeepLab models"""

import copy
import time
from typing import Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.nn.functional import one_hot
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from tqdm import tqdm

from models import DeepLabWrapper


class Trainer:
    """This class trains DeepLab models given a configuration of hyperparameters"""

    def __init__(
        self,
        deeplab: DeepLabWrapper,
        dataloaders: torch.utils.data.DataLoader,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        num_epochs: int = 25,
        logger=None,
        save_model_path: str = None,
    ):
        """Initialization method for Trainer base class

        Args:
            deeplab: (DeepLabWrapper)
                DeepLab model to train
            dataloaders: (torch.utils.data.DataLoader)
                Dataloaders for training and validation
            criterion: (torch.nn.CrossEntropyLoss)
                Loss function to use for training
            optimizer: (torch.optim.Adam)
                Optimizer to use for training
            num_epochs: (int, optional)
                Number of epochs to train the model for
            logger: (optional)
                Logger to use for logging training metrics
            save_model_path: (str, optional)
                Path to save the trained model
        """
        self.deeplab = deeplab
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.logger = logger
        self.save_model_path = save_model_path

    def train(self) -> Tuple[DeepLabWrapper, list]:
        """This function is used to train a model

        Returns:
            model, val_mean_iou_history
        """
        since = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        val_mean_iou_history = []
        best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())
        best_mean_iou = 0.0
        self.deeplab.model.to(device)

        scaler = GradScaler()

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 10)

            for phase in ["train", "valid"]:
                if phase == "train":
                    self.deeplab.model.train()
                else:
                    self.deeplab.model.eval()

                mean_iou = MeanIoU(
                    num_classes=self.deeplab.num_mask_channels,
                    include_background=False,
                ).to(device)
                gds = GeneralizedDiceScore(
                    num_classes=self.deeplab.num_mask_channels,
                    include_background=False,
                ).to(device)
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
                        with autocast(device_type="cuda", dtype=torch.float16):
                            outputs = self.deeplab.model(inputs)
                            loss = self.criterion(outputs["out"], labels)
                        preds = torch.argmax(outputs["out"], dim=1)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            scaler.scale(loss).backward()
                            scaler.step(self.optimizer)
                            scaler.update()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    mean_iou.update(preds, labels)
                    gds.update(
                        one_hot(
                            preds,
                            num_classes=self.deeplab.num_mask_channels,
                        ).permute(0, 3, 1, 2),
                        one_hot(
                            labels,
                            num_classes=self.deeplab.num_mask_channels,
                        ).permute(0, 3, 1, 2),
                    )
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_mean_iou = mean_iou.compute().item()
                epoch_gds = gds.compute().item()

                if self.logger:
                    self.logger.log(
                        {
                            f"{phase}_loss": epoch_loss,
                            f"{phase}_mean_iou": epoch_mean_iou,
                            f"{phase}_gds": epoch_gds,
                            "epoch": epoch + 1,
                        }
                    )

                print(f"{phase} Loss: {epoch_loss:.4f} mIoU: {epoch_mean_iou:.4f} GDS: {epoch_gds:.4f}")
                if phase == "valid":
                    val_mean_iou_history.append(epoch_mean_iou)

                    if epoch_mean_iou > best_mean_iou:
                        best_mean_iou = epoch_mean_iou
                        best_model_wts = copy.deepcopy(self.deeplab.model.state_dict())
            print()

        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best val mean IoU: {best_mean_iou:4f}")

        # load best model weights
        self.deeplab.model.load_state_dict(best_model_wts)

        # save the model
        model_path = self.save_model_path or f"runs/{self.deeplab.backbone}_v1.{self.num_epochs}.pt"
        self.deeplab.save_model(model_path)

        if self.logger:
            self.logger.finish()

        return self.deeplab, val_mean_iou_history
