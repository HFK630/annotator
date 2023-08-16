from annotate import ImageAnnotator
from image_resizer import ImageResizer
import os
import cv2
import json

APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_PATH = os.path.join(APP_PATH, 'images')
PRE_ANNOTATED_IMAGE_FOLDER = os.path.join(IMAGES_PATH, 'pre_annotated_images')
ANNOTATED_IMAGE_FOLDER = os.path.join(IMAGES_PATH, 'annotated_images')
RESIZED_IMAGE_HEIGHT = 720
RESIZED_IMAGE_WIDTH = 1280


def main():
    for image_name in os.listdir(PRE_ANNOTATED_IMAGE_FOLDER):
        image_path = os.path.join(PRE_ANNOTATED_IMAGE_FOLDER, image_name)
        resized_image = ImageResizer().resize(image_path, RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)
        annotator = ImageAnnotator(image_path, resized_image)
        save = annotator.annotate()

        # Save the annotated image
        if save:
            annotated_image_path = os.path.join(ANNOTATED_IMAGE_FOLDER, image_name)
            cv2.imwrite(annotated_image_path, annotator.img)
        
            # Save the annotation file
            annotation_file_path = os.path.join(ANNOTATED_IMAGE_FOLDER, image_name.split('.')[0] + '.json')
            with open(annotation_file_path, 'w') as file:
                json.dump(annotator.annotation, file)


if __name__ == "__main__":
    main()
    