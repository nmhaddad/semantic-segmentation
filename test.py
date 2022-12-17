""" Runs inference on single image inputs """

import os.path as op

import numpy as np
from PIL import Image
import yaml

from models import DeepLabWrapper
from utils import display_example_pair, vis_segmentation, vis_grid_4x3


if __name__ == '__main__':

    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    example_image = Image.open(op.join(config['DATA_PATH'],
                                       'train/iid000008/rgb.jpg'))
    example_mask = Image.open(op.join(config['DATA_PATH'],
                                      'train/iid000008/labels.png'))
    image_display = np.array(example_image)
    mask_display = np.array(example_mask.convert('RGB'))
    display_example_pair(image_display, mask_display)
    w, h, *_ = example_image.size
    model = DeepLabWrapper(model_path=config['LOAD_MODEL_PATH'], input_shape=(w,h))

    predicted_masks = model.process(example_image)

    vis_segmentation(example_image, np.array(predicted_masks))
    vis_grid_4x3(model, config['DATA_PATH'])
