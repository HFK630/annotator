import cv2
import numpy as np


class ImageResizer:
    """Class to resize images to fit the screen."""

    def resize(self, image_path: str, wanted_width: int, wanted_height: int) -> np.ndarray:
        """Resize the image to the specified width and height."""
        original_img = cv2.imread(image_path)
        height, width = original_img.shape[:2]
        scale = min(wanted_width / width, wanted_height / height)
        new_width, new_height = int(width * scale), int(height * scale)

        # Resize the original image
        resized_img = cv2.resize(original_img, (new_width, new_height))

        # Create a white canvas of size wanted_width x wanted_height
        img_original = np.ones(
            (wanted_height, wanted_width, 3), dtype=np.uint8) * 255

        start_x = (wanted_width - new_width) // 2
        start_y = (wanted_height - new_height) // 2
        img_original[start_y:start_y + new_height,
                     start_x:start_x + new_width] = resized_img
        return img_original
