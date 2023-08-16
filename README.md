# Image Annotator

Image Annotator is a powerful tool designed to annotate images with bounding boxes. It provides features like zooming, dragging, drawing rectangles, and more. The application follows SOLID principles, making it highly maintainable and extensible.

## Features

- **Image Resizing:** Automatically resizes images to fit the screen.
- **Bounding Box Annotation:** Allows users to draw bounding boxes around objects in the image.
- **Zooming:** Users can zoom in and out of the image for precise annotations.
- **Dragging:** Enables dragging the image for better positioning.
- **Undo/Redo:** Provides undo and redo functionality for the bounding boxes.
- **Save/Exit Options:** Users can save their annotations or exit without saving.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/annotator.git
   ```
2. Navigate to the project directory:
    ```bash
    cd annotator
    ```
3. (Optional) Create a venv and activate it:
    ```bash
    python -m venv venv
    # For windows
    .\venv\Scripts\activate
    # For Linux
    source venv\bin\activate
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Controls

- **Click and hold Left Mouse Button:** Draw bounding boxes.
- **Click and hold Middle Mouse Button:** Drag the image.
- **Scroll Mouse Wheel:** Zoom in/out.
- **Ctrl + z:** Undo the last bounding box.
- **Ctrl + y:** Redo the last bounding box.
- **press q on the keyboard:** Saves the current picture and annotaitions and goes to the next one (untill it iterates over all of them)
- **Press Esc on the keyboard:** Exits the current picture and annotaitions without saving it and goes to the next one (untill it iterates over all of them)

## Contributing

Contributions are welcome! Please read the CONTRIBUTING.md for details on how to contribute.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

