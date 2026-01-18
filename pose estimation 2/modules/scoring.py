import numpy as np
import math

class ScoreCalculator:
    def __init__(self):
        pass

    # --- [ALGORITMA] COSINE SIMILARITY (WIRAGA) ---
    def calculate_cosine_similarity(self, v1, v2):
        """
        Menghitung kemiripan dua vektor menggunakan Cosine Similarity.
        Output 1.0 = Sangat Mirip, 0.0 = Tegak Lurus, -1.0 = Berlawanan.
        """
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
            
        return dot_product / (norm_v1 * norm_v2)

    def get_body_angles(self, lm_list):
        """Mengubah koordinat tubuh menjadi vektor sudut penting"""
        if len(lm_list) < 33:
            return None
        
        p11 = np.array([lm_list[11][1], lm_list[11][2]])
        p13 = np.array([lm_list[13][1], lm_list[13][2]])
        p23 = np.array([lm_list[23][1], lm_list[23][2]])
        
        vektor_lengan = p13 - p11
        vektor_badan = p23 - p11
        
        return vektor_lengan, vektor_badan

    # --- [ALGORITMA] TIMING MATCHING (WIRAMA) ---
    def calculate_score(self, lm_guru, lm_siswa):
        if not lm_guru or not lm_siswa:
            return 0, 0

        v_lengan_g, v_badan_g = self.get_body_angles(lm_guru)
        v_lengan_s, v_badan_s = self.get_body_angles(lm_siswa)
        
        similarity = self.calculate_cosine_similarity(v_lengan_g, v_lengan_s)
        
        score_postur = max(0, (similarity + 1) / 2 * 100)

        if score_postur > 80:
            score_timing = 100
        elif score_postur > 60:
            score_timing = 70
        else:
            score_timing = 40
            
        return score_postur, score_timing