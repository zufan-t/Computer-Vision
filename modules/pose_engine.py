import cv2
import mediapipe as mp
import numpy as np
import os

class PoseEngine:
    def __init__(self, model_path=r"C:\PKMKC\data\pose_landmarker_full.task"):
        # Initialize MediaPipe Tasks
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create the landmarker with video mode
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO)
        
        try:
            self.landmarker = PoseLandmarker.create_from_options(options)
            print("[INFO] Model MediaPipe dimuat sukses.")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model: {e}")
            raise e

    def calculate_angle(self, a, b, c):
        """Menghitung sudut antara 3 titik (a, b=pusat, c)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
        return angle

    def extract_features_from_video(self, video_path):
        """
        Membaca video dan mengembalikan list vektor fitur per frame.
        Fitur: [Sudut Lutut Kiri, Sudut Lutut Kanan, Sudut Siku Kiri, Sudut Siku Kanan]
        """
        cap = cv2.VideoCapture(video_path)
        features = []
        
        if not cap.isOpened():
            print(f"[ERROR] Tidak bisa membuka video: {video_path}")
            return np.array([])

        print(f"[INFO] Memproses video: {video_path}...")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 30.0
            print("[WARN] FPS tidak terdeteksi, menggunakan default 30.0")
        
        # Ensure distinct integer timestamps
        frame_interval_ms = 1000.0 / fps
        frame_idx = 0

        print(f"[INFO] FPS: {fps}, Interval: {frame_interval_ms:.2f}ms")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # MediaPipe Tasks requires RGB image as mp.Image
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Calculate timestamp in ms (must be strictly increasing)
            # Use frame_idx to avoid accumulation errors
            current_timestamp_ms = int(frame_idx * frame_interval_ms)
            frame_idx += 1
            
            try:
                detection_result = self.landmarker.detect_for_video(mp_image, current_timestamp_ms)
            except Exception as e:
                print(f"[WARN] Frame {frame_idx-1} skipped (ts={current_timestamp_ms}ms): {e}")
                continue

            if detection_result.pose_landmarks:
                # Ambil orang pertama yang terdeteksi
                landmarks = detection_result.pose_landmarks[0]
                
                # Helper function untuk ambil (x,y)
                get_coord = lambda idx: [landmarks[idx].x, landmarks[idx].y]
                
                # Index MediaPipe: 
                # 11=BahuKi, 12=BahuKa, 13=SikuKi, 14=SikuKa, 15=PergelanganKi, 16=PergelanganKa
                # 23=PinggulKi, 24=PinggulKa, 25=LututKi, 26=LututKa, 27=EngkelKi, 28=EngkelKa
                
                # 1. Sudut Mendak (Lutut)
                angle_knee_l = self.calculate_angle(get_coord(23), get_coord(25), get_coord(27))
                angle_knee_r = self.calculate_angle(get_coord(24), get_coord(26), get_coord(28))
                
                # 2. Sudut Tangan (Siku)
                angle_elbow_l = self.calculate_angle(get_coord(11), get_coord(13), get_coord(15))
                angle_elbow_r = self.calculate_angle(get_coord(12), get_coord(14), get_coord(16))
                
                features.append([angle_knee_l, angle_knee_r, angle_elbow_l, angle_elbow_r])
                
        cap.release()
        return np.array(features)