""" This file is used to process and save video data. """

import numpy as np
import yaml
import cv2

from utils import draw_segmentation, save_video
from models import DeepLabWrapper


def process_video(model_path: str, data_path: str) -> None:
    """ Processes a video with a given model

    Args:
        model_path: (str)
            path of model to use for processing
        data_path: (str)
            path of video to use for processing

    Returns:
        None
    """
    video = cv2.VideoCapture(data_path)
    model = DeepLabWrapper(model_path = model_path, input_shape=(int(video.get(3)), int(video.get(4))))
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        predicted_masks = model.process(frame)
        overlay = draw_segmentation(frame, np.array(predicted_masks))
        frames.append(overlay)

    if frames:
        save_video(frames, config['SAVE_VIDEO'])

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    with open('config/video_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    process_video(config['LOAD_MODEL_PATH'], config['DATA_PATH'])
