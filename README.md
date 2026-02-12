# Hand Gesture Recognition using Pure OpenCV (TinyML Approach)

## Project Description
This project implements a lightweight real-time hand gesture recognition system using OpenCV.  
The system detects gestures and maps them to actions like Light ON/OFF.

## Technologies Used
- Python 3.12
- OpenCV
- NumPy

## Features
- Real-time gesture detection
- Convex hull & contour-based finger counting
- Edge-device compatible (TinyML concept)

## How to Run

1. Install requirements:
   pip install opencv-python numpy

2. Run:
   python gesture_detection.py

## Output Gestures
- Fist → Light OFF
- One Finger → Light ON
- Two Fingers → 2 Fingers
- Open Hand → 5 Fingers
