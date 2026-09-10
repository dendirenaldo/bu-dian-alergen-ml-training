"""Image preprocessing for OCR pipeline."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_image(path: str) -> np.ndarray:
    """Preprocess an image for OCR.

    Applies grayscale conversion, bilateral filter, Otsu thresholding,
    foreground/background normalization, and morphological closing.

    Args:
        path: Path to the image file.

    Returns:
        Preprocessed binary image as a numpy array.

    Raises:
        ValueError: If the image cannot be read.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Normalize foreground/background for more stable OCR
    if np.mean(thresh) < 127:
        thresh = 255 - thresh

    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    logger.debug("Preprocessed image: %s", path)
    return processed


def load_image_paths(dataset_dir: str, extensions: set[str] | None = None) -> list[str]:
    """Load all image file paths from a directory.

    Args:
        dataset_dir: Path to the dataset directory.
        extensions: Set of allowed file extensions (e.g. {'.jpg', '.png'}).

    Returns:
        Sorted list of image file paths.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    paths = sorted(
        str(p)
        for p in dataset_path.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    logger.info("Found %d images in %s", len(paths), dataset_dir)
    return paths
