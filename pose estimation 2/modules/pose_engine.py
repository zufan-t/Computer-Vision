import mediapipe as mp
import cv2
import numpy as np

class PoseEngine:
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True, 
                 enable_segmentation=False, smooth_segmentation=True,
                 detection_con=0.5, track_con=0.5):
        
        self.mode = mode
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.complexity,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=detection_con,
            min_tracking_confidence=track_con
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # --- [ALGORITMA] VARIABLE UNTUK EXPONENTIAL MOVING AVERAGE (EMA) ---
        self.prev_landmarks = None
        self.alpha = 0.6  # Faktor Halus (0.0 - 1.0). 0.6 = Stabil.

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        
        lm_list = []
        if self.results.pose_landmarks:
            # --- [ALGORITMA] PROSES PENGHALUSAN (SMOOTHING) ---
            # Kita manipulasi data sebelum digambar
            if self.prev_landmarks is None:
                self.prev_landmarks = self.results.pose_landmarks
            else:
                for i, landmark in enumerate(self.results.pose_landmarks.landmark):
                    prev = self.prev_landmarks.landmark[i]
                    
                    # RUMUS EMA: New = (Current * alpha) + (Previous * (1-alpha))
                    landmark.x = (landmark.x * self.alpha) + (prev.x * (1 - self.alpha))
                    landmark.y = (landmark.y * self.alpha) + (prev.y * (1 - self.alpha))
                    landmark.z = (landmark.z * self.alpha) + (prev.z * (1 - self.alpha))
                
                # Simpan untuk frame depan
                self.prev_landmarks = self.results.pose_landmarks

            # Ekstrak data ke list biar mudah dihitung
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy, lm.z, lm.visibility])

            if draw:
                self.mp_drawing.draw_landmarks(
                    img, 
                    self.results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                )
                
        return img, lm_list