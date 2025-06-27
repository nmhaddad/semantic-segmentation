"""Wrapper for torchvision DeepLabv3 models"""

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabWrapper(pl.LightningModule):
    """Wrapper used to add additional features and methods to torchvision DeepLabv3 models

    Attributes:
        model: torchvision.models.segmentation.DeepLabv3
            wrapped model
        backbone: str
            name of backbone
        num_mask_channels: int
            Number of output classes
        model_path: str
            Path to pretrained model to load
        preprocess_transform: torchvision.transform
            transforms to apply to input images during inference
        parameters:
            model parameters
    """

    def __init__(
        self,
        backbone=None,
        num_mask_channels=None,
        model_path=None,
    ):
        """Initializes a DeepLabWrapper object

        Args:
            backbone: str, optional
                Which backbone to load. Options: mobilenetv3large, resnet50, resnet101
            num_mask_channels: int, optional
                number of classes to predict
            model_path: str, optional
                Path for custom pretrained models
        """
        super().__init__()
        self.model = None
        self.backbone = backbone
        self.num_mask_channels = num_mask_channels
        self.model_path = model_path
        self.preprocess_transform = None

        if self.model_path:
            self.load_model()
        else:
            self.initialize_model()

        if not self.model:
            raise RuntimeError("Couldn't create model with given configuration")

        self.cuda = torch.cuda.is_available()

        self.parameters = self.model.parameters()

        if self.cuda:
            self.model.to("cuda")

    def load_model(self, eval: bool = True) -> None:
        """Loads a model from a file

        Args:
            eval: (bool, optional)
                Sets model to evaluation mode for inference

        Returns:
            None
        """
        self.model = torch.load(self.model_path, weights_only=False)
        if eval:
            self.model.eval()
        self.preprocess_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def save_model(self, model_path: str) -> None:
        """Saves model to the given model path

        Args:
            model_path: (str)
                Path to save the model to

        Returns:
            None
        """
        torch.save(self.model, model_path)

    def initialize_model(self):
        """Initializes a DeepLabv3 model from the torchvision package


        Returns:
            None
        """
        match self.backbone.lower():
            case "resnet101":
                self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
                self.model.classifier = DeepLabHead(2048, self.num_mask_channels)
            case "resnet50":
                self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
                self.model.classifier = DeepLabHead(2048, self.num_mask_channels)
            case "mobilenetv3large":
                self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
                self.model.classifier = DeepLabHead(960, self.num_mask_channels)
            case _:
                raise ValueError(
                    "Unknown backbone selected in configuration. Please select from RESNET50, RESNET101, or MOBILENETV3LARGE"
                )

    def preprocess(self, image: np.ndarray | Image.Image) -> torch.Tensor:
        """Preprocesses input into format required for processing"""
        # apply the same transforms that were applied to input images when training the model
        input_tensor: torch.Tensor = self.preprocess_transform(image)
        # put the image in a batch (as expected by the model)
        input_batch: torch.Tensor = input_tensor.unsqueeze(0)
        # move the input and model to GPU for speed if available
        if self.cuda:
            input_batch = input_batch.to("cuda")
        return input_batch

    def forward(self, image: np.ndarray | Image.Image) -> Image.Image:
        """Processes input through a DeepLabv3 model"""
        input_batch = self.preprocess(image)
        with torch.no_grad():
            output: torch.Tensor = self.model(input_batch)["out"][0]
        output_predictions = output.argmax(0)
        return Image.fromarray(output_predictions.byte().cpu().numpy())
