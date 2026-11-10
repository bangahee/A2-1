"""Prepare a 64×64 grayscale image from Matplotlib's bundled public sample."""

from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
from matplotlib import cbook, image as mpl_image


def prepare_image():
    """Read sample pixels, convert RGB to gray in NumPy, and area-average."""
    source = cbook.get_sample_data("grace_hopper.jpg", asfileobj=False)
    rgb = np.asarray(mpl_image.imread(source), dtype=float) / 255.0
    gray = rgb[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    # Center-crop the largest square divisible by 64, then average blocks.
    side = min(gray.shape) // 64 * 64
    top, left = (gray.shape[0] - side) // 2, (gray.shape[1] - side) // 2
    crop = gray[top:top + side, left:left + side]
    block = side // 64
    small = crop.reshape(64, block, 64, block).mean(axis=(1, 3))
    (ROOT / "data").mkdir(exist_ok=True)
    np.save(ROOT / "data/grace_hopper_64.npy", small, allow_pickle=False)
    return small


if __name__ == "__main__":
    prepare_image()
