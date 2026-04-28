# Gesture Engine

A real-time hand gesture recognition system that uses MediaPipe to detect hand landmarks and translate gestures into computer actions. Control your computer via hand gestures captured through your webcam.

## Features

- **Real-time Hand Detection** - Continuously detects hand landmarks from webcam feed
- **Gesture Recognition** - Identifies different hand gestures and maps them to actions
- **Multi-threaded Processing** - Separates frame capture, processing, and display for smooth performance
- **Voice Feedback** - Provides audio feedback for detected gestures and actions
- **Cross-platform Desktop Control** - Integrates with system to perform mouse and keyboard actions
- **Logging** - Records all detected gestures and actions to a log file

## Requirements

- Python 3.8+
- Webcam
- Windows/macOS/Linux

## Installation

1. **Clone or download the project:**
   ```bash
   cd d:\real_project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MediaPipe Hand Landmarker model:**
   - Download `hand_landmarker.task` from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   - Place it in the project root directory

## Usage

Run the gesture control application:

```bash
python main.py
```

The application will:
1. Access your default webcam
2. Display a video feed with hand landmarks overlaid
3. Detect your hand gestures in real-time
4. Execute corresponding actions
5. Press **'q'** in the video window to quit

## Project Structure

```
gesture_engine.py      # Core gesture detection and processing logic
main.py                # Application entry point and UI display
hand_landmarker.task   # MediaPipe hand detection model
requirements.txt       # Python package dependencies
gesture_log.txt        # Log file for detected gestures and actions
README.md             # This file
```

## Components

### GestureEngine (`gesture_engine.py`)
- Initializes MediaPipe HandLandmarker model
- Processes video frames to detect hand landmarks
- Identifies hand gestures based on landmark positions
- Triggers actions based on recognized gestures
- Logs all activities and provides voice feedback

### GestureApp (`main.py`)
- Manages camera capture and video display
- Implements multi-threaded architecture:
  - **Capture Thread** - Continuously reads frames from webcam
  - **Processing Thread** - Detects gestures using GestureEngine
  - **Main Thread** - Displays annotated frames and FPS counter
- Handles exit signals and cleanup

## Gesture Recognition

The system recognizes various hand gestures based on finger positions and orientations:
- Finger extension states are determined by comparing tip vs. PIP (proximal interphalangeal) landmark positions
- Gestures are mapped to specific computer actions
- A cooldown period prevents rapid subsequent actions

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Video capture and frame processing |
| `mediapipe` | Hand landmark detection |
| `pyautogui` | Desktop automation (mouse/keyboard control) |
| `pyttsx3` | Text-to-speech voice feedback |

## Performance Optimization

- **Queue-based frame handling** - Drops old frames if processing falls behind
- **Multi-threading** - Separates I/O, processing, and display operations
- **Video mode** - Uses MediaPipe's video processing mode for frame continuity
- **FPS monitoring** - Displays real-time performance metrics

## Logging

All detected gestures and performed actions are logged to `gesture_log.txt` with timestamps for debugging and analysis.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Could not open default camera" | Ensure webcam is connected and not in use by another application |
| Low gesture recognition accuracy | Ensure good lighting and place hand fully in frame |
| Model file not found | Download `hand_landmarker.task` and place in project root |
| Slow performance | Reduce model confidence thresholds or close other applications |

## Future Enhancements

- Support for two-hand gestures
- Custom gesture training
- Gesture profiles for different applications
- Configuration file for customizable keybindings

## License

This project uses MediaPipe (Apache License 2.0)

## Support

For issues or questions, check the gesture_log.txt file for debugging information.
