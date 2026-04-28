import cv2
import threading
import queue
import time
import sys
from gesture_engine import GestureEngine

class GestureApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # Verify camera is accessible
        if not self.cap.isOpened():
            print("Error: Could not open default camera (index 0).")
            sys.exit(1)
            
        self.engine = GestureEngine()
        
        # We only really need the latest frame, so queue maxsize 1 is good
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        self.running = False
        
        # Event for exit signal from processing thread
        self.exit_event = threading.Event()
        
        # Latest data to display on UI
        self.latest_frame = None
        self.current_gesture = "None"
        self.current_action = "None"
        
    def _capture_thread(self):
        """Thread to capture frames from camera as fast as possible"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Flip frame horizontally for a more intuitive selfie-view
            frame = cv2.flip(frame, 1)
            
            # Put frame in queue for processing, drop if queue is full
            try:
                self.frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
                
    def _processing_thread(self):
        """Thread to process frames with MediaPipe"""
        while self.running:
            try:
                # Wait for a frame to process
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            # Process frame
            annotated_frame, gesture, action, exit_flag = self.engine.process_frame(frame)
            if exit_flag:
                self.exit_event.set()
            
            # Send result back
            try:
                # If queue is full (display hasn't caught up), drop the old result
                if self.result_queue.full():
                    self.result_queue.get_nowait()
                self.result_queue.put_nowait((annotated_frame, gesture, action))
            except queue.Full:
                pass

    def run(self):
        self.running = True
        
        # Start threads
        capture_thread = threading.Thread(target=self._capture_thread)
        capture_thread.daemon = True
        capture_thread.start()
        
        process_thread = threading.Thread(target=self._processing_thread)
        process_thread.daemon = True
        process_thread.start()
        
        print("Starting gesture control application. Press 'q' to quit in the video window.")
        
        start_time = time.time()
        fps_counter = 0
        fps = 0
        
        try:
            while self.running:
                # Check for exit signal
                if self.exit_event.is_set():
                    self.running = False
                    break
                
                # Get the latest processed result for display
                try:
                    display_frame, gesture, action = self.result_queue.get(timeout=0.05)
                    self.current_gesture = gesture
                    if action != "None":
                        self.current_action = action
                except queue.Empty:
                    # If no processed frame is ready, just continue
                    display_frame = None
                
                if display_frame is not None:
                    fps_counter += 1
                    current_time = time.time()
                    if current_time - start_time > 1.0:
                        fps = fps_counter
                        fps_counter = 0
                        start_time = current_time
                        
                    # --- UI Overlay ---
                    # Transparent black background for text
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
                    display_frame = cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0)
                    
                    # Update text colors and status based on cooldown
                    status_text = ""
                    action_color = (0, 255, 0) # Default green

                    time_since_last_action = current_time - self.engine.last_action_time
                    if time_since_last_action < self.engine.cooldown:
                        status_text = f" (Cooldown: {self.engine.cooldown - time_since_last_action:.1f}s)"
                        action_color = (0, 255, 255) # Yellow when active/cooling down
                    else:
                        self.current_action = "Waiting..."
                        action_color = (150, 150, 150) # Gray when waiting

                    # Display the current action and gesture mappings
                    display_text = f"Action: {self.current_action}"

                    cv2.putText(display_frame, f"FPS: {fps}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Gesture: {self.current_gesture}", (20, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(display_frame, display_text, (20, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)

                    # Add gesture-to-action mappings for clarity
                    mappings = [
                        "Thumbs Up: Volume Up",
                        "Thumbs Down: Volume Down",
                        "Open Palm: Google Meet Mute",
                        "OK Sign: Video Toggle",
                        "Victory: Screenshot",
                        "Point: Exit Program"
                    ]
                    for i, mapping in enumerate(mappings):
                        cv2.putText(display_frame, mapping, (20, 160 + i * 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                                
                    cv2.imshow("Gesture Controller", display_frame)
                    
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
        finally:
            self.running = False
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = GestureApp()
    app.run()
