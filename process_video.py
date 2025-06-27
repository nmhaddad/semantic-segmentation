"""This file is used to process and save video data."""

import cv2
import numpy as np
import yaml
from PIL import Image

from models import DeepLabWrapper
from utils import overlay_mask_cv2


def process_video(model_path: str, data_path: str, outfile_path: str) -> None:
    """Processes a video with a given model

    Args:
        model_path: (str)
            path of model to use for processing
        data_path: (str)
            path of video to use for processing
        outfile_path: (str)
            path to save the processed video

    Returns:
        None
    """
    video = cv2.VideoCapture(data_path)
    out = cv2.VideoWriter(outfile_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (513, 513))
    model = DeepLabWrapper(model_path=model_path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # resize and crop the input frame (returns frame for building video)
        pil_frame = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB))
        resized_frame = model.resize_and_crop_input(pil_frame)
        predicted_mask = model(resized_frame)
        masked_image = overlay_mask_cv2(np.array(resized_frame), np.array(predicted_mask))
        out.write(masked_image)

    video.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    with open("config/video_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    process_video(config["LOAD_MODEL_PATH"], config["DATA_PATH"], config["SAVE_VIDEO"])
