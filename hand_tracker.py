import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os


class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.model_path = self._get_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.max_hands = max_hands
        
    def _get_model_path(self):
        # Download model if not exists
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            print(f"Downloading hand tracking model...")
            urllib.request.urlretrieve(url, model_path)
            print(f"Model saved to {model_path}")
        return model_path
        
    def find_hands(self, frame, draw=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.results = self.detector.detect(mp_image)
        
        if self.results.hand_landmarks and draw:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_landmarks(frame, hand_landmarks)
        return frame
    
    def _draw_landmarks(self, frame, hand_landmarks):
        # Draw hand connections manually since we don't have mp_draw
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17), (0, 17)  # Palm
        ]
        h, w, _ = frame.shape
        
        for idx, landmark in enumerate(hand_landmarks):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        
        for start, end in connections:
            if start < len(hand_landmarks) and end < len(hand_landmarks):
                x1, y1 = int(hand_landmarks[start].x * w), int(hand_landmarks[start].y * h)
                x2, y2 = int(hand_landmarks[end].x * w), int(hand_landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    def get_landmark_positions(self, frame):
        positions = []
        if self.results.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in self.results.hand_landmarks:
                landmarks = {}
                for id, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks[id] = (cx, cy)
                positions.append(landmarks)
        return positions
    
    def is_pinching(self, landmarks):
        if landmarks is None or 4 not in landmarks or 8 not in landmarks:
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        return distance < 40
    
    def get_index_finger_tip(self, landmarks):
        if landmarks and 8 in landmarks:
            return landmarks[8]
        return None
    
    def is_index_finger_extended(self, landmarks):
        """Check if index finger is extended (tip above PIP joint) for single-finger drawing"""
        if landmarks is None:
            return False
        
        # Check if we have all necessary landmarks
        # 8 = index tip, 6 = index PIP (joint below tip)
        # We also check that middle finger (12) is NOT extended to isolate index finger
        if 8 not in landmarks or 6 not in landmarks:
            return False
        
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        
        # Check index finger is extended (tip is higher/further up than PIP)
        # Note: y coordinate increases downward in image coordinates
        index_extended = index_tip[1] < index_pip[1]
        
        # Optional: check that other fingers are curled (not extended) to isolate single finger
        # This prevents accidental drawing when all fingers are up
        other_fingers_curled = True
        finger_checks = [
            (12, 10),  # middle: tip vs PIP
            (16, 14),  # ring
            (20, 18)   # pinky
        ]
        
        for tip_id, pip_id in finger_checks:
            if tip_id in landmarks and pip_id in landmarks:
                if landmarks[tip_id][1] < landmarks[pip_id][1]:  # finger is extended
                    other_fingers_curled = False
                    break
        
        return index_extended and other_fingers_curled
    
    def release(self):
        pass
