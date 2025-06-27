"""Wrapper for torchvision DeepLabv3 models"""

from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.transforms import v2


class DeepLabWrapper(pl.LightningModule):
    """Wrapper for torchvision DeepLabv3 models"""

    def __init__(
        self,
        backbone: Optional[str] = None,
        num_mask_channels: Optional[int] = None,
        model_path: Optional[str] = None,
    ):
        """Initializes a DeepLabWrapper instance

        Args:
            backbone: (str, optional)
                Name of the backbone to use for the model. If None, a model will be initialized with the default backbone.
            num_mask_channels: (int, optional)
                Number of output channels for the segmentation mask. If None, defaults to 21 (for COCO dataset).
            model_path: (str, optional)
                Path to a pretrained model to load. If None, a new model will be initialized.
        """
        super().__init__()
        self.backbone = backbone
        self.num_mask_channels = num_mask_channels
        self.model_path = model_path

        self.model = None
        self.input_transform = None
        self.transform = None

        if self.model_path:
            self.load_model()
        else:
            self.initialize_model()

        self.create_transform()
        self.cuda = torch.cuda.is_available()
        self.parameters = self.model.parameters()
        if self.cuda:
            self.model.to("cuda")

    def resize_and_crop_input(
        self, image: Image.Image, mask: Optional[Image.Image] = None
    ) -> Image.Image | Tuple[Image.Image, Image.Image]:
        """Resizes and crops input image to the required size for the model

        Args:
            image: (PIL.Image)
                Input image to resize and crop
            mask: (PIL.Image, optional)
                Mask to resize and crop. If None, only the image is resized and cropped.

        Returns:
            PIL.Image: resized and cropped image (and mask if provided)
        """
        if not self.input_transform:
            self.create_input_transform(image.size)

        if mask is not None:
            return self.input_transform(image, mask)
        return self.input_transform(image)

    def load_model(self, eval: bool = True) -> None:
        """Loads a model from a file

        Args:
            eval: (bool, optional)
                Sets model to evaluation mode for inference
        """
        self.model = torch.load(self.model_path, weights_only=False)
        if eval:
            self.model.eval()

    def create_input_transform(self, input_shape: Tuple[int, int]) -> None:
        """Creates the input transform used to resize and crop input images

        Args:
            input_shape: (Tuple[int, int])
                Shape of the input image (height, width)

        """
        # resize image so shortest side is 513px
        w, h = input_shape
        if w < h:
            new_w = 513
            new_h = int(h * (513 / w))
        else:
            new_h = 513
            new_w = int(w * (513 / h))
        self.input_transform = v2.Compose(
            [
                v2.Resize((new_h, new_w)),
                v2.CenterCrop((513, 513)),
            ]
        )

    def create_transform(self) -> None:
        """Creates the transforms used to preprocess input images"""
        self.transform = v2.Compose(
            [
                v2.Resize((513, 513)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def save_model(self, model_path: str) -> None:
        """Saves model to the given model path

        Args:
            model_path: (str)
                Path to save the model to
        """
        torch.save(self.model, model_path)

    def initialize_model(self) -> None:
        """Initializes a DeepLabv3 model from the torchvision package"""
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

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses input into format required for processing"""
        # apply the same transforms that were applied to input images when training the model (training-serving skew)
        input_tensor: torch.Tensor = self.transform(image)
        input_batch: torch.Tensor = input_tensor.unsqueeze(0)
        if self.cuda:
            input_batch = input_batch.to("cuda")
        return input_batch

    def forward(self, image: Image.Image) -> Image.Image:
        """Processes input through a DeepLabv3 model"""
        input_batch = self.preprocess(image)
        with torch.no_grad():
            output: torch.Tensor = self.model(input_batch)["out"][0]
        # get the output predictions
        output_predictions = output.argmax(0)
        return Image.fromarray(output_predictions.byte().cpu().numpy())
