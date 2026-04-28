import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import time
import threading
import os
import math
import pyttsx3

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

class GestureEngine:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}\n"
                "Download it from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        self.last_action_time = 0
        self.cooldown = 1.5
        self.current_gesture = "None"
        self.current_action = "None"

        self.voice_engine = pyttsx3.init()
        self.voice_engine.setProperty('rate', 150)

        self.log_file = open("gesture_log.txt", "a")

    def process_frame(self, frame):
        """Processes a frame, detects gesture, triggers actions, returns annotated frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.frame_timestamp_ms += 33
        result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)

        self.current_gesture = "None"

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks)
                self.current_gesture = self._detect_gesture(hand_landmarks)
                exit_flag = self._trigger_action(self.current_gesture)
                if exit_flag:
                    return frame, self.current_gesture, self.current_action, True

        return frame, self.current_gesture, self.current_action, False

    def _draw_landmarks(self, frame, landmarks):
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for p in points:
            cv2.circle(frame, p, 4, (0, 255, 0), -1)
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 1)

    def _landmark_distance(self, lm, idx_a, idx_b):
        """Euclidean distance between two landmarks in normalised coords."""
        a = lm[idx_a]
        b = lm[idx_b]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _hand_scale(self, lm):
        """Reference scale: wrist (0) to middle-finger MCP (9).
        Keeps the pinch threshold proportional regardless of camera distance."""
        return self._landmark_distance(lm, 0, 9)

    def _detect_gesture(self, landmarks):
        """
        Gesture → Action mapping
        ─────────────────────────────────────────
        Thumbs Up   → Volume Up
        Thumbs Down → Volume Down
        Open Palm   → Google Meet mute toggle
        OK Sign     → Video toggle (Ctrl+Shift+V)
        Victory     → Screenshot
        Point       → Exit program
        """
        lm = landmarks

        # ── Finger extension: tip y < PIP y means the finger is raised ───────
        index_up  = lm[8].y  < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up   = lm[16].y < lm[14].y
        pinky_up  = lm[20].y < lm[18].y

        # Thumb uses vertical position relative to its own joints
        thumb_tip_y = lm[4].y
        thumb_ip_y  = lm[3].y   # IP joint (first knuckle from tip)
        thumb_mcp_y = lm[2].y   # MCP joint (base knuckle)

        thumb_pointing_up   = thumb_tip_y < thumb_ip_y  - 0.02
        thumb_pointing_down = thumb_tip_y > thumb_mcp_y + 0.02

        fingers_curled = not index_up and not middle_up and not ring_up and not pinky_up

        # ── Thumbs Up ─────────────────────────────────────────────────────────
        if fingers_curled and thumb_pointing_up:
            return "Thumbs Up"

        # ── Thumbs Down ───────────────────────────────────────────────────────
        if fingers_curled and thumb_pointing_down:
            return "Thumbs Down"

        # ── Fist (all curled, thumb neutral) ──────────────────────────────────
        if fingers_curled:
            return "Fist"

        # ── Open Palm ─────────────────────────────────────────────────────────
        if index_up and middle_up and ring_up and pinky_up:
            return "Open Palm"

        # ── OK Sign ───────────────────────────────────────────────────────────
        # Conditions:
        #   • middle, ring, pinky are extended (up)
        #   • index is NOT fully extended (bent to form the circle)
        #   • thumb tip and index tip are close together (pinch)
        # The threshold (0.32) is relative to hand scale so distance to camera
        # doesn't matter.
        if middle_up and ring_up and pinky_up and not index_up:
            pinch_dist = self._landmark_distance(lm, 4, 8)
            scale      = self._hand_scale(lm)
            if scale > 0 and (pinch_dist / scale) < 0.32:
                return "OK Sign"

        # ── Victory / Peace sign ──────────────────────────────────────────────
        if index_up and middle_up and not ring_up and not pinky_up:
            return "Victory"

        # ── Point (only index up) ─────────────────────────────────────────────
        if index_up and not middle_up and not ring_up and not pinky_up:
            return "Point"

        return "Unknown"

    def _trigger_action(self, gesture):
        current_time = time.time()
        if current_time - self.last_action_time < self.cooldown:
            return False

        action_triggered = False
        voice_message    = ""

        # ── Thumbs Up → Volume Up ─────────────────────────────────────────────
        if gesture == "Thumbs Up":
            print(f"[{time.strftime('%H:%M:%S')}] Increasing Volume")
            threading.Thread(target=self._execute_shortcut, args=(['volumeup'],)).start()
            self.current_action = "Volume Up"
            voice_message       = "Volume up"
            action_triggered    = True

        # ── Thumbs Down → Volume Down ─────────────────────────────────────────
        elif gesture == "Thumbs Down":
            print(f"[{time.strftime('%H:%M:%S')}] Decreasing Volume")
            threading.Thread(target=self._execute_shortcut, args=(['volumedown'],)).start()
            self.current_action = "Volume Down"
            voice_message       = "Volume down"
            action_triggered    = True

        # ── Open Palm → Google Meet mute toggle ───────────────────────────────
        elif gesture == "Open Palm":
            print(f"[{time.strftime('%H:%M:%S')}] Toggling Mute in Google Meet")
            threading.Thread(target=self._toggle_google_meet_mute).start()
            self.current_action = "Google Meet Mute Toggle"
            voice_message       = "Mute toggled in Google Meet"
            action_triggered    = True

        # ── OK Sign → Video Toggle ────────────────────────────────────────────
        elif gesture == "OK Sign":
            print(f"[{time.strftime('%H:%M:%S')}] Triggering Video Toggle (Ctrl+Shift+V)")
            threading.Thread(target=self._execute_shortcut, args=(['ctrl', 'shift', 'v'],)).start()
            self.current_action = "Video Toggle"
            voice_message       = "Video toggle"
            action_triggered    = True

        # ── Victory → Screenshot ──────────────────────────────────────────────
        elif gesture == "Victory":
            print(f"[{time.strftime('%H:%M:%S')}] Triggering Screenshot")
            threading.Thread(target=self._take_screenshot).start()
            self.current_action = "Screenshot"
            voice_message       = "Screenshot taken"
            action_triggered    = True

        # ── Point → Exit ──────────────────────────────────────────────────────
        elif gesture == "Point":
            print(f"[{time.strftime('%H:%M:%S')}] Exiting program")
            self.current_action = "Exit"
            voice_message       = "Exiting program"
            self.last_action_time = current_time
            self.log_file.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Exit gesture detected\n"
            )
            self.log_file.flush()
            threading.Thread(target=self._speak, args=(voice_message,)).start()
            return True  # Signal caller to quit

        if action_triggered:
            self.last_action_time = current_time
            self.log_file.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {gesture}: {self.current_action}\n"
            )
            self.log_file.flush()
            threading.Thread(target=self._speak, args=(voice_message,)).start()

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _execute_shortcut(self, keys):
        try:
            pyautogui.hotkey(*keys)
        except Exception as e:
            print(f"Error executing shortcut {keys}: {e}")

    def _take_screenshot(self):
        try:
            filename = f"screenshot_{int(time.time())}.png"
            pyautogui.screenshot(filename)
            print(f"Screenshot saved as {filename}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")

    def _speak(self, message):
        try:
            self.voice_engine.say(message)
            self.voice_engine.runAndWait()
        except Exception as e:
            print(f"Error speaking: {e}")

    def _toggle_google_meet_mute(self):
        """Ctrl+D is the default Google Meet mute shortcut."""
        try:
            pyautogui.hotkey('ctrl', 'd')
        except Exception as e:
            print(f"Error toggling mute in Google Meet: {e}")