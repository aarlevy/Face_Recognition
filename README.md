# Face Recognition System

A simple and efficient face recognition system built with Python. This system can recognize faces in real-time using your webcam and match them against a database of known faces.

## Features

- Real-time face detection and recognition
- Support for multiple image formats (JPG, JPEG, PNG, HEIC)
- Confidence score display
- Detailed logging
- Easy to use interface

## Requirements

```bash
opencv-python
numpy
face_recognition
Pillow
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/aarlevy/Face_Recognition.git
cd Face_Recognition
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Create a directory called `Faces` in the project root
2. Inside `Faces`, create a subdirectory for each person you want to recognize
3. Add photos of each person to their respective directories
4. Run the program:
```bash
python face_recognition.py
```

## Directory Structure

```
Face_Recognition/
│
├── face_recognition.py    # Main program
├── Faces/                 # Directory containing face database
│   ├── Person1/          # Directory for first person
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── Person2/          # Directory for second person
│       ├── photo1.jpg
│       └── photo2.jpg
└── requirements.txt       # Package dependencies
```

## Notes

- For best results, use clear, well-lit photos for the face database
- The system uses a confidence threshold of 0.5 for recognition
- Press 'q' to quit the program while running

## License

MIT License 