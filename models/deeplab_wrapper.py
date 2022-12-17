""" Wrapper for torchvision DeepLabv3 models """

import numpy as np
import pytorch_lightning as pl
from torchvision import models, transforms
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch
from PIL import Image


class DeepLabWrapper(pl.LightningModule):
    """ Wrapper used to add additional features and methods to torchvision DeepLabv3 models

    Attributes:
        model: torchvision.models.segmentation.DeepLabv3
            wrapped model
        backbone: str
            name of backbone
        num_mask_channels: int
            Number of output classes
        model_path: str
            Path to pretrained model to load
        input_shape: Tuple[int, int]
        input_width: int
            Input width property
        input_height: int
            Input height property
        preprocess_transform: torchvision.transform
            transforms to apply to input images during inference
        parameters:
            model parameters
    """

    def __init__(self, backbone=None, num_mask_channels=None, input_shape=None, model_path=None, pretrained: bool=True, progress: bool=True, aux_loss: bool=True):
        """ Initializes a DeepLabWrapper object

        Args:
            backbone: str, optional
                Which backbone to load. Options: mobilenetv3large, resnet50, resnet101
            num_mask_channels: int, optional
                number of classes to predict
            input_shape: Tuple[int, int], optional
                Input shape (width, height)
            model_path: str, optional
                Path for custom pretrained models
            pretrained: bool, optional
                Whether or not to use built-in pretrained weights
            progress: bool, optional
                Debug output for training
            aux_loss: bool, optional
                Use auxiliary loss when training Inception models
        """
        super().__init__()
        self.model = None
        self.backbone = backbone
        self.num_mask_channels = num_mask_channels
        self.model_path = model_path
        self.input_shape = input_shape
        self.preprocess_transform = None

        if self.model_path:
            self.load_model()
        else:
            self.initialize_model(pretrained, progress, aux_loss)

        if not self.model:
            raise RuntimeError("Couldn't create model with given configuration")

        self.cuda = torch.cuda.is_available()

        self.parameters = self.model.parameters()

        if self.cuda:
            self.model.to('cuda')


    @property
    def input_width(self) -> int:
        """ Gets input width

        Returns:
            width: (int)
                width of input images
        """
        return self.input_shape[0]

    @property
    def input_height(self) -> int:
        """ Gets input height

        Returns:
            height: (int)
                height of input images
        """
        return self.input_shape[1]

    def load_model(self, eval: bool = True) -> None:
        """ Loads a model from a file

        Args:
            eval: (bool, optional)
                Sets model to evaluation mode for inference

        Returns:
            None
        """
        self.model = torch.load(self.model_path)
        if eval:
            self.model.eval()
        self.preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def save_model(self, model_path: str) -> None:
        """ Saves model to the given model path

        Args:
            model_path: (str)
                Path to save the model to

        Returns:
            None
        """
        torch.save(self.model, model_path)

    def initialize_model(self, pretrained: bool, progress: bool, aux_loss: bool):
        """ Initializes a DeepLabv3 model from the torchvision package

        Args:
            pretrained: bool
                Use a pretrained backbone
            progress:
                Show download progress
            aux_loss:
                Use auxiliary loss during training

        Returns:
            None
        """
        if self.backbone == 'resnet101':
            model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=progress, aux_loss=aux_loss)
            model.classifier = DeepLabHead(2048, self.num_mask_channels)
        elif self.backbone == 'resnet50':
            model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained, progress=progress, aux_loss=aux_loss)
            model.classifier = DeepLabHead(2048, self.num_mask_channels)
        elif self.backbone == 'mobilenetv3large':
            model = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained, progress=progress, aux_loss=aux_loss)
            model.classifier = DeepLabHead(960, self.num_mask_channels)
        else:
            raise ValueError('Unknown backbone selected in configuration. Please select from RESNET50, RESNET101, or MOBILENETV3LARGE')
        self.model = model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ Preprocesses input into format required for processing """
        # apply the same transforms that were applied to input images when training the model
        input_tensor = self.preprocess_transform(image)
        # put the image in a batch (as expected by the model)
        input_batch = input_tensor.unsqueeze(0)
        # move the input and model to GPU for speed if available
        if self.cuda:
            input_batch = input_batch.to('cuda')
        return input_batch

    def process(self, image: np.ndarray) -> np.ndarray:
        """ Processes input through a DeepLabv3 model """
        input_batch = self.preprocess(image)
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        # TODO only numpy arrays!
        return Image.fromarray(output_predictions.byte().cpu().numpy()).resize((self.input_width, self.input_height))
