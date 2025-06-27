"""Utility file containing a variety of helper functions"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec, patches
from matplotlib.colors import ListedColormap, to_rgb

LABEL_NAMES = [
    "unknown",
    "non-traversable",
    "rough trail",
    "smooth trail",
    "traversable grass",
    "low vegetation",
    "obstacle",
    "high vegetation",
    "sky",
]
COLORS = [
    "#000000",  # unknown - black
    "#8B4513",  # non-traversable - brown
    "#D2691E",  # rough trail - chocolate
    "#F4A460",  # smooth trail - sandy brown
    "#90EE90",  # traversable grass - light green
    "#228B22",  # low vegetation - forest green
    "#FF0000",  # obstacle - red
    "#006400",  # high vegetation - dark green
    "#87CEEB",  # sky - sky blue
]

RGB_COLORS = [tuple(int(c * 255) for c in to_rgb(h)) for h in COLORS]


def vis_segmentation(image: np.ndarray, mask: np.ndarray) -> None:
    """Visualizes input image, segmentation map and overlay view

    Args:
        image: (np.ndarray)
            the rgb image
        mask: (np.ndarray)
            the mask of the input image
    """
    cmap = ListedColormap(COLORS[: len(LABEL_NAMES)])

    plt.figure(figsize=(20, 5))
    grid_spec = gridspec.GridSpec(
        1,
        4,
        width_ratios=[6, 6, 6, 4],
    )

    # input image
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input Image")
    # mask
    plt.subplot(grid_spec[1])
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(LABEL_NAMES) - 1)
    plt.axis("off")
    plt.title("Mask")
    # overlay
    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=len(LABEL_NAMES) - 1, alpha=0.5)
    plt.axis("off")
    plt.title("Mask Overlay")
    # legend
    legend_elements = []
    for i, (label, color) in enumerate(zip(LABEL_NAMES, COLORS[: len(LABEL_NAMES)])):
        legend_elements.append(patches.Rectangle((0, 0), 1, 1, facecolor=color, label=f"{i}: {label}"))
    plt.subplot(grid_spec[3])
    plt.legend(
        handles=legend_elements, loc="center", frameon=False, title="Legend", title_fontsize="large", fontsize="large"
    )
    plt.axis("off")

    plt.grid("off")
    plt.savefig("segmentation_visualization.png", bbox_inches="tight", pad_inches=0.1)


def overlay_mask_cv2(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Visualizes input image overlayed with segmentation map

    Args:
        image: (np.ndarray)
            the rgb image
        mask: (np.ndarray)
            the mask to overlay

    Returns:
        np.ndarray: the image with the mask overlayed
    """

    # Create color mask image
    color_mask = np.zeros_like(image, dtype=np.uint8)
    for idx, rgb in enumerate(RGB_COLORS):
        color_mask[mask == idx] = rgb

    # Ensure input is uint8
    image = image.astype(np.uint8)

    # convert from RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    # write the image and mask to debug
    cv2.imwrite("debug_image.png", image)
    cv2.imwrite("debug_mask.png", color_mask)

    # Blend
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def display_example_pair(image: np.ndarray, mask: np.ndarray) -> None:
    """Visualizes input image and segmentation map. Used for visualizations.

    Args:
        image: (np.ndarray) the rgb image
        mask: (np.ndarray) the mask of the input image
    """
    _, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(image)
    ax[0].axis("off")
    ax[0].set_title("Original Image")
    ax[1].imshow(mask)
    ax[1].axis("off")
    ax[1].set_title("Mask")
