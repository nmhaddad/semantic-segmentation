"""YamahaCMU Dataloaders"""

import glob
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2


class YamahaCMUDataset(VisionDataset):
    """A class that represents the Yamaha-CMU Off-Road dataset

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

    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        """Initializes a YamahaCMUDataset object

        Args:
            root: (str)
                the root directory
            transforms: (Optional[Callable])
                torch transforms to use
        """
        super().__init__(root, transforms)
        self.image_paths = []
        self.mask_paths = []
        image_mask_pairs = glob.glob(root + "/*/")
        for image_mask in image_mask_pairs:
            self.image_paths.append(glob.glob(f"{image_mask}*.jpg")[0])
            self.mask_paths.append(glob.glob(f"{image_mask}*.png")[0])

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the item at the given index of this dataset

        Args:
            index: (int)
                the index of the item to get

        Returns:
            the sample at the given index
        """
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        image = tv_tensors.Image(image, dtype=torch.float32)
        mask = tv_tensors.Mask(mask, dtype=torch.long)

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


def get_dataloader(data_dir: str, batch_size: int = 2) -> Dict[str, DataLoader]:
    """Creates a dataloader for the given dataset

    Args:
        data_dir: (str)
            the directory of the dataset
        batch_size: (int=2)
            the batch size to use

    Returns:
        torch.utils.data.DataLoader
    """

    transforms = v2.Compose(
        [
            v2.ToTensor(),
            v2.RandomCrop(513),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_datasets = {x: YamahaCMUDataset(data_dir + x, transforms=transforms) for x in ["train", "valid"]}

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=4)
        for x in ["train", "valid"]
    }
    return dataloaders
