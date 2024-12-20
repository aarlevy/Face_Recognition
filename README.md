# Face Recognition Attendance System

A modern face recognition attendance system using deep learning, suitable for schools and businesses. The system uses the VGG-Face model for face recognition and maintains an attendance log in CSV format.

## Features

- Real-time face recognition using webcam
- Multiple face detection and recognition
- Attendance logging with timestamp
- Support for multiple people
- CSV-based attendance records
- High accuracy using deep learning
- User-friendly interface

## Setup

1. Make sure you have Python 3.8+ installed
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

Place reference face images in the `Ref-Faces` directory with the following structure:
```
Ref-Faces/
├── Person1_Name/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Person2_Name/
│   ├── image1.jpg
│   └── ...
└── ...
```

## Usage

1. Add reference faces:
   - Create a folder with the person's name under `Ref-Faces`
   - Add 2-3 clear face photos of the person in their folder
   - Photos should show the face clearly without obstruction

2. Run the system:
   ```bash
   python attendance_system.py
   ```

3. The system will:
   - Load and register all faces from the `Ref-Faces` directory
   - Start the webcam for real-time recognition
   - Record attendance in `attendance_log.csv`

4. Press 'q' to quit the program

## Attendance Log

The system creates an `attendance_log.csv` file with the following information:
- Name
- Date
- Time
- Status

## Notes

- Ensure good lighting for better recognition
- Face should be clearly visible to the camera
- Multiple people can be recognized simultaneously
- Each person's attendance is logged only once per day

## License

MIT License 