import cv2
import os
import json
import numpy as np
from typing import Any, List, Optional, Tuple, Dict


class ImageAnnotator:
    """Class to annotate images with bounding boxes."""

    def __init__(
        self,
        img_path: str,
        img: np.ndarray,
        darken_by: float = 0.5
    ) -> None:
        self.img_original = img
        self.height, self.width = self.img_original.shape[:2]
        self.img = self.img_original.copy()
        self.img_temp = self.img.copy()
        self.annotation: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]] = {
            'image': os.path.basename(img_path), 'boxes': []}
        self.start_x, self.start_y = 0, 0
        self.drag_start_x, self.drag_start_y = 0, 0
        self.zoom_scale = 1.0
        self.last_created_box = []
        self.darken_by = darken_by
        self.dragging = False
        self.drawing = False
        self.top_left_pt = (-1, -1)
        self.bottom_right_pt = (-1, -1)

    def _reset_image_size(self) -> None:
        """Reset the image size to the original size and removes text from the image."""
        self.zoom_scale = 1.0
        self.start_x, self.start_y = 0, 0
        self.update_image(display_text=False)

    def undo(self) -> None:
        """Undo the latest created box."""
        if self.last_created_box:
            self.annotation["boxes"].remove(self.last_created_box[0])
            self.update_image()
        else:
            print("No boxes to undo")

    def redo(self) -> None:
        """Redo the latest created box."""
        if self.last_created_box:
            if self.last_created_box[0] not in self.annotation["boxes"]:
                self.annotation["boxes"].append(self.last_created_box[0])
                self.update_image()
            else:
                raise ValueError("Box already exists")
        else:
            print("No boxes to redo")

    def handle_wheel_event(self, x: int, y: int, flags: int) -> None:
        """Zoom in or out of the image."""
        delta = -1 if flags < 0 else 1
        self.zoom_scale += delta * 0.1
        self.zoom_scale = max(1.0, self.zoom_scale)  # Limit zoom out

        # Update the center of zoom based on the current mouse position
        x_center = self.start_x + x
        y_center = self.start_y + y

        self.update_image(x_center, y_center)

    def preform_drag(self, x: int, y: int) -> None:
        """Drag the image."""
        delta_x = (x - self.drag_start_x)
        delta_y = (y - self.drag_start_y)

        self.start_x -= delta_x
        self.start_y -= delta_y

        self.drag_start_x, self.drag_start_y = x, y
        self.update_image()

    def draw_rectangle(self, x: int, y: int) -> None:
        """Draw a rectangle on the image."""
        if self.drawing:
            mask = np.ones_like(self.img_temp) * (1 - self.darken_by)
            cv2.rectangle(mask, self.top_left_pt, (x, y), (1, 1, 1), -1)
            self.img_temp = (self.img_temp * mask).astype(np.uint8)
            cv2.rectangle(self.img_temp, self.top_left_pt,
                          (x, y), (0, 255, 0), 2)
        cv2.line(self.img_temp, (0, y),
                 (self.img_temp.shape[1], y), (255, 0, 0), 1)
        cv2.line(self.img_temp, (x, 0),
                 (x, self.img_temp.shape[0]), (255, 0, 0), 1)

    def set_rectangle(self, x: int, y: int) -> None:
        """Set the rectangle on the image."""
        self.drawing = False
        self.bottom_right_pt = (x, y)
        top_left_original = tuple(
            (np.array(self.top_left_pt) + [self.start_x, self.start_y]) / self.zoom_scale)
        bottom_right_original = tuple(
            (np.array(self.bottom_right_pt) + [self.start_x, self.start_y]) / self.zoom_scale)
        annotation_box = (top_left_original, bottom_right_original)
        self.annotation['boxes'].append(annotation_box)
        self.last_created_box = [annotation_box]
        self.update_image()  # Update the image without moving it.

    def mouse_actions(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: Optional[Any]
    ) -> None:
        """Mouse actions on the image."""
        self.img_temp = self.img.copy()

        if event == cv2.EVENT_MOUSEWHEEL:
            self.handle_wheel_event(x, y, flags)

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.drag_start_x, self.drag_start_y = x, y

        # Handle the movement while dragging
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.preform_drag(x, y)

        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging = False

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.top_left_pt = (x, y)
            self.update_image()

        elif event == cv2.EVENT_MOUSEMOVE:
            self.draw_rectangle(x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.set_rectangle(x, y)

    def update_image(
        self,
        x_center: Optional[int] = None,
        y_center: Optional[int] = None,
        display_text: bool = True
    ) -> None:
        """Updates the image according to the zoom scale and the cropping coordinates.
        Args:
            x_center (int, optional): The x coordinate of the center of the zoom.
            y_center (int, optional): The y coordinate of the center of the zoom.
            display_text (bool, optional): Whether to display the number of boxes on the image. Defaults to True.
        Returns:
            None
        If x_center or y_center are not given, the image will stay still and not move."""
        
        if x_center is None:
            x_center = self.start_x + self.width // 2
        if y_center is None:
            y_center = self.start_y + self.height // 2

        new_height, new_width = int(
            self.height * self.zoom_scale), int(self.width * self.zoom_scale)

        # Resizing the original image according to the zoom scale
        zoomed_image = cv2.resize(self.img_original, (new_width, new_height))

        # Calculating the cropping coordinates
        self.start_x = min(max(x_center - self.width //
                           2, 0), new_width - self.width)
        self.start_y = min(max(y_center - self.height //
                           2, 0), new_height - self.height)
        end_x = self.start_x + self.width
        end_y = self.start_y + self.height

        # Cropping the zoomed image to the window size
        cropped_image = zoomed_image[self.start_y:end_y, self.start_x:end_x]

        # Create a white canvas of size 1280x720
        self.img = np.ones(
            (self.height, self.width, 3), dtype=np.uint8) * 255

        # Paste the cropped zoomed image onto the canvas
        self.img[:cropped_image.shape[0],
                 :cropped_image.shape[1]] = cropped_image

        # Redraw the bounding boxes
        for box in self.annotation['boxes']:
            scaled_top_left = tuple(
                (np.array(box[0]) * self.zoom_scale - [self.start_x, self.start_y]).astype(int))
            scaled_bottom_right = tuple(
                (np.array(box[1]) * self.zoom_scale - [self.start_x, self.start_y]).astype(int))
            cv2.rectangle(self.img, scaled_top_left,
                          scaled_bottom_right, (0, 255, 0), 2)

        box_count_text = f'Boxes: {len(self.annotation["boxes"])}'
        if display_text:
            cv2.putText(self.img, box_count_text, (10, 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 2)

        self.img_temp = self.img.copy()

    def annotate(self) -> bool:
        """Display the annotations on the image.
        
        Returns:
            bool: True if the image needs to be saved, False otherwise."""
        cv2.namedWindow('Annotate')
        cv2.setMouseCallback('Annotate', self.mouse_actions)
        while True:
            cv2.imshow('Annotate', self.img_temp)
            key = cv2.waitKey(1)
            if key == ord('q'):
                self._reset_image_size()  # Reset the image size before saving
                return True  # Save the image
            elif key == 27:  # ASCII for esc
                return False  # Don't save the image
            elif key == 26:  # ASCII for ctrl+z
                try:
                    self.undo()
                except ValueError:
                    print("Can't undo anymore")
            elif key == 25:  # ASCII for ctrl+y
                try:
                    self.redo()
                except ValueError:
                    print("Can't redo anymore")
