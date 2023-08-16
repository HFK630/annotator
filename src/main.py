from annotate import ImageAnnotator
import os
import cv2
import json

APP_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_PATH = os.path.join(APP_PATH, 'images')
PRE_ANNOTATED_IMAGE_FOLDER = os.path.join(IMAGES_PATH, 'pre_annotated_images')
ANNOTATED_IMAGE_FOLDER = os.path.join(IMAGES_PATH, 'annotated_images')


def main():
    for filename in os.listdir(PRE_ANNOTATED_IMAGE_FOLDER):
        filepath = os.path.join(PRE_ANNOTATED_IMAGE_FOLDER, filename)
        annotator = ImageAnnotator(filepath)
        save = annotator.annotate()

        # Save the annotated image
        if save:
            annotated_image_path = os.path.join(ANNOTATED_IMAGE_FOLDER, filename)
            cv2.imwrite(annotated_image_path, annotator.img)
        
            # Save the annotation file
            annotation_file_path = os.path.join(ANNOTATED_IMAGE_FOLDER, filename.split('.')[0] + '.json')
            with open(annotation_file_path, 'w') as file:
                json.dump(annotator.annotation, file)


if __name__ == "__main__":
    main()
    