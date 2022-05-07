import torch
import numpy as np
from PIL import Image

from utils import display_example_pair, run_inference, vis_segmentation, vis_grid_4x3

if __name__ == '__main__':
    
    example_image = Image.open('data/yamaha_v0/train/iid000008/rgb.jpg')
    example_mask = Image.open('data/yamaha_v0/train/iid000008/labels.png')
    image_display = np.array(example_image)
    mask_display = np.array(example_mask.convert('RGB'))
    display_example_pair(image_display, mask_display)

    model = torch.load('models/mobilenet_model_v0.2.pt')
    model.eval()
    predicted_masks = run_inference(model, example_image)

    vis_segmentation(example_image, np.array(predicted_masks))

    vis_grid_4x3(model)