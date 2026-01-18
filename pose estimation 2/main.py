import cv2
import time
import os
from modules.pose_engine import PoseEngine
from modules.scoring import ScoreCalculator

def main():
    # 1. SETUP PATH VIDEO
    # Pastikan file ini ada di folder 'data'
    path_guru = "data/guru.mp4"   
    path_siswa = "data/siswa.mp4" # Nanti bisa diganti 0 untuk Webcam
    
    # 2. OUTPUT VIDEO SETUP
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "hasil_analisis_sitari.mp4")

    # 3. INISIALISASI
    cap_guru = cv2.VideoCapture(path_guru)
    cap_siswa = cv2.VideoCapture(path_siswa)
    
    # Panggil Class Engine (Otak Deteksi)
    detector_guru = PoseEngine(complexity=1) # Guru model ringan
    detector_siswa = PoseEngine(complexity=1) # Siswa model ringan
    
    # Panggil Class Scoring (Otak Hitung)
    scorer = ScoreCalculator()

    # Siapkan Video Writer
    # Kita akan gabungkan video guru (kiri) dan siswa (kanan)
    w_guru = int(cap_guru.get(3))
    h_guru = int(cap_guru.get(4))
    # Resize agar sama tinggi (misal 720p)
    target_h = 480
    target_w = int((w_guru / h_guru) * target_h)
    
    # Output canvas size (2 video berdampingan)
    final_w = target_w * 2
    final_h = target_h
    
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (final_w, final_h))

    print(f"Mulai Analisis... Hasil akan disimpan di {save_path}")

    while True:
        # Baca Frame
        success_g, img_guru = cap_guru.read()
        success_s, img_siswa = cap_siswa.read()
        
        # Loop video guru jika habis (biar siswa bisa terus latihan)
        if not success_g:
            cap_guru.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success_g, img_guru = cap_guru.read()

        # Stop jika video siswa habis
        if not success_s:
            break

        # Resize agar seragam
        img_guru = cv2.resize(img_guru, (target_w, target_h))
        img_siswa = cv2.resize(img_siswa, (target_w, target_h))

        # --- PROSES COMPUTER VISION ---
        # 1. Deteksi Guru (Tanpa gambar skeleton biar bersih, ambil datanya saja)
        img_guru, lm_guru = detector_guru.find_pose(img_guru, draw=False) 
        
        # 2. Deteksi Siswa (Gambar skeleton + Smoothing otomatis jalan di dalam)
        img_siswa, lm_siswa = detector_siswa.find_pose(img_siswa, draw=True)

        # 3. Hitung Skor
        nilai_postur = 0
        nilai_timing = 0
        
        if len(lm_guru) != 0 and len(lm_siswa) != 0:
            nilai_postur, nilai_timing = scorer.calculate_score(lm_guru, lm_siswa)

        # --- VISUALISASI HASIL (UI) ---
        # Gabungkan gambar berdampingan
        img_final = cv2.hconcat([img_guru, img_siswa])
        
        # Gambar Kotak Nilai
        cv2.rectangle(img_final, (10, 10), (250, 120), (0, 0, 0), cv2.FILLED)
        cv2.putText(img_final, f"POSTUR (Wiraga)", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(img_final, f"{int(nilai_postur)}%", (20, 80), cv2.FONT_HERSHEY_BOLD, 3, (0, 255, 255), 3)
        
        cv2.putText(img_final, f"TIMING (Wirama)", (20, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        
        # Indikator Warna (Merah/Hijau)
        warna_timing = (0, 255, 0) if nilai_timing > 80 else (0, 0, 255)
        cv2.putText(img_final, f"{int(nilai_timing)}%", (20, 190), cv2.FONT_HERSHEY_BOLD, 3, warna_timing, 3)

        # Simpan frame ke video output
        out.write(img_final)

        # Tampilkan di layar
        cv2.imshow("SITARI - Smart Dance Evaluator", img_final)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_guru.release()
    cap_siswa.release()
    out.release()
    cv2.destroyAllWindows()
    print("Selesai! Cek folder output.")

if __name__ == "__main__":
    main()