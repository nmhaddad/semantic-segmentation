"""Runs inference on single image inputs"""

import os.path as op

import numpy as np
import yaml
from PIL import Image

from models import DeepLabWrapper
from utils import vis_segmentation

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

image = Image.open(op.join(config["DATA_PATH"], "train/iid000008/rgb.jpg"))
mask = Image.open(op.join(config["DATA_PATH"], "train/iid000008/labels.png"))

model = DeepLabWrapper(model_path=config["LOAD_MODEL_PATH"])
# dynamically resize and crop the input image to the required size for the model
image, mask = model.resize_and_crop_input(image, mask)
predicted_mask = model(image)
vis_segmentation(image, np.array(predicted_mask))
