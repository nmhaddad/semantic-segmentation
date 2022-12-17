""" YamahaCMU Dataloaders"""

import glob
from typing import Any, Callable, Optional

import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import numpy as np
import cv2
from PIL import Image


class YamahaCMUDataset(VisionDataset):
    """ A class that represents the Yamaha-CMU Off-Road dataset

    Attributes:
        root: (str)
            the root directory
        transforms: (Optional[Callable])
            torch transforms to use

    Methods:
        __len__():
            returns the length of the dataset
        __getitem__(index):
            returns the item at the given index of this dataset
    """

    def __init__(self, root: str, resize_shape: tuple,
                 transforms: Optional[Callable] = None) -> None:
        """ Initializes a YamahaCMUDataset object

        Args:
            root: (str)
                the root directory
            transforms: (Optional[Callable])
                torch transforms to use
        """
        super().__init__(root, transforms)
        image_paths = []
        mask_paths = []
        image_mask_pairs = glob.glob(root + '/*/')
        for image_mask in image_mask_pairs:
            image_paths.append(glob.glob(image_mask + '*.jpg')[0])
            mask_paths.append(glob.glob(image_mask + '*.png')[0])
        self.image_names = image_paths
        self.mask_names = mask_paths

        if resize_shape:
            self.image_height, self.image_width = resize_shape
            self.resize = True
        else:
            self.image_height, self.image_width = (544, 1024)
            self.resize = False

    def __len__(self) -> int:
        """ Returns the length of the dataset """
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        """ Returns the item at the given index of this dataset

        Args:
            index: (int)
                the index of the item to get

        Returns:
            the sample at the given index
        """
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        image = Image.open(image_path)
        image = image.convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        class_colors = np.unique(mask)
        if self.resize:
            mask = cv2.resize(mask,
                              dsize=(self.image_width, self.image_height),
                              interpolation=cv2.INTER_CUBIC)
        # remove void class (atv)
        if 0 in class_colors:
            class_colors = class_colors[1:]
        label_masks = mask == class_colors[:, None, None]
        masks = np.zeros((8, self.image_height, self.image_width))
        for index, class_color in enumerate(class_colors):
            masks[class_color - 1] = label_masks[index, :, :] * 255
        sample = {"image": image, "mask": masks}
        if self.transforms:
            sample["image"] = self.transforms(sample["image"])
            sample['mask'] = torch.as_tensor(sample['mask'], dtype=torch.uint8)
        return sample


def get_dataloader(data_dir: str, batch_size: int=2, resize_shape: tuple=None) -> torch.utils.data.DataLoader:
    """ Creates a dataloader for the given dataset

    Args:
        data_dir: (str)
            the directory of the dataset
        batch_size: (int=2)
            the batch size to use

    Returns:
        torch.utils.data.DataLoader
    """

    if resize_shape:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    image_datasets = {
        x: YamahaCMUDataset(data_dir + x, resize_shape, transforms=preprocess) for x in ['train', 'valid']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, drop_last=True)
        for x in ['train', 'valid']
    }
    return dataloaders
