import os.path as op

import torch
import numpy as np
from PIL import Image
import yaml

from utils import (display_example_pair, run_inference, vis_segmentation,
                   vis_grid_4x3)


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

    model = torch.load(config['LOAD_MODEL_PATH'])
    model.eval()

    predicted_masks = run_inference(model, example_image)

    vis_segmentation(example_image, np.array(predicted_masks))
    vis_grid_4x3(model, config['DATA_PATH'])
