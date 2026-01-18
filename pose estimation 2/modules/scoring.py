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
        # Contoh: Mengambil sudut ketiak (Bahu-Siku vs Bahu-Pinggul)
        # Index MediaPipe: 11=BahuKiri, 13=SikuKiri, 23=PinggulKiri
        if len(lm_list) < 33:
            return None
        
        # Ambil koordinat penting (x, y)
        p11 = np.array([lm_list[11][1], lm_list[11][2]]) # Bahu Kiri
        p13 = np.array([lm_list[13][1], lm_list[13][2]]) # Siku Kiri
        p23 = np.array([lm_list[23][1], lm_list[23][2]]) # Pinggul Kiri
        
        # Buat Vektor Lengan dan Badan
        vektor_lengan = p13 - p11
        vektor_badan = p23 - p11
        
        return vektor_lengan, vektor_badan

    # --- [ALGORITMA] TIMING MATCHING (WIRAMA) ---
    # *Catatan: DTW asli berat untuk real-time, ini implementasi "Windowed Matching"*
    # Kita mengecek apakah pose siswa saat ini cocok dengan pose guru 
    # di rentang waktu +/- 0.5 detik.
    def calculate_score(self, lm_guru, lm_siswa):
        if not lm_guru or not lm_siswa:
            return 0, 0

        # 1. HITUNG SKOR POSTUR (Menggunakan Cosine Similarity)
        v_lengan_g, v_badan_g = self.get_body_angles(lm_guru)
        v_lengan_s, v_badan_s = self.get_body_angles(lm_siswa)
        
        # Hitung kemiripan vektor lengan guru vs lengan siswa
        similarity = self.calculate_cosine_similarity(v_lengan_g, v_lengan_s)
        
        # Konversi -1..1 menjadi nilai 0..100
        # Jika sim=1 (0 derajat beda) -> Nilai 100
        # Jika sim=0 (90 derajat beda) -> Nilai 50
        score_postur = max(0, (similarity + 1) / 2 * 100)

        # 2. HITUNG SKOR TIMING (Sederhana)
        # Jika Postur bagus (>80) artinya timing pas.
        # Jika Postur jelek, mungkin telat.
        # (Disini nanti logika DTW kompleks diletakkan)
        if score_postur > 80:
            score_timing = 100 # On beat
        elif score_postur > 60:
            score_timing = 70  # Sedikit meleset
        else:
            score_timing = 40  # Miss beat
            
        return score_postur, score_timing