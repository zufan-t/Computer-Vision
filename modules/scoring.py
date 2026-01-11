import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, master_features, student_features):
        print("[INFO] Menjalankan Dynamic Time Warping (DTW)...")
        
        if len(master_features) == 0 or len(student_features) == 0:
            return {
                "score_sequence": 0,
                "score_posture": 0,
                "final_score": 0
            }

        # 1. SEQUENCE MATCHING (DTW)
        distance, path = fastdtw(master_features, student_features, dist=euclidean)
        
        # Normalisasi jarak
        normalized_distance = distance / len(path)
        
        # Skor Sequence (Semakin kecil distance, semakin bagus)
        score_sequence = max(0, 100 - (normalized_distance * 2))

        # 2. POSTURE ACCURACY
        total_error = 0
        count = 0
        
        for idx_master, idx_student in path:
            vec_master = master_features[idx_master]
            vec_student = student_features[idx_student]
            
            # Hitung selisih rata-rata di frame tersebut
            frame_error = np.mean(np.abs(vec_master - vec_student))
            total_error += frame_error
            count += 1
            
        avg_posture_error = total_error / count if count > 0 else 100
        score_posture = max(0, 100 - avg_posture_error)

        return {
            "dtw_distance": normalized_distance,
            "avg_angle_error": avg_posture_error,
            "score_sequence": score_sequence,
            "score_posture": score_posture,
            "final_score": (0.4 * score_sequence) + (0.6 * score_posture)
        }