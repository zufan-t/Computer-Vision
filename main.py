import os
import numpy as np
from modules.pose_engine import PoseEngine
from modules.scoring import Evaluator

def main():
    # --- KONFIGURASI PATH (Menggunakan Raw String 'r') ---
    # Sesuaikan path ini dengan komputer Anda
    path_master = r"C:\PKMKC\data\video\Denok_master.mp4"
    path_siswa = r"C:\PKMKC\data\video\Denok_murid.mp4"
    cache_master = r"C:\PKMKC\data\cache\master_data.npy"
    
    # Inisialisasi Modul
    engine = PoseEngine()
    evaluator = Evaluator()

    # --- LANGKAH 1: PROSES DATA MASTER ---
    if os.path.exists(cache_master):
        print("[INFO] Memuat data Master dari cache...")
        master_features = np.load(cache_master)
    else:
        print("[INFO] Cache tidak ditemukan. Memproses video Master...")
        master_features = engine.extract_features_from_video(path_master)
        
        # Buat folder cache jika belum ada
        os.makedirs(os.path.dirname(cache_master), exist_ok=True)
        
        if len(master_features) > 0:
            np.save(cache_master, master_features)
            print("[INFO] Data Master berhasil disimpan ke cache.")
        else:
            print("[ERROR] Gagal memproses Master. Cek path video.")
            return

    # --- LANGKAH 2: PROSES DATA SISWA ---
    print("[INFO] Memproses video Siswa...")
    student_features = engine.extract_features_from_video(path_siswa)
    
    if len(student_features) == 0:
        print("[ERROR] Gagal memproses Siswa. Cek path video.")
        return

    # --- LANGKAH 3: EVALUASI ---
    print(f"[INFO] Frame Master: {len(master_features)} | Frame Siswa: {len(student_features)}")
    
    result = evaluator.evaluate(master_features, student_features)

    # --- LANGKAH 4: LAPORAN ---
    print("\n" + "="*40)
    print("      RAPOR EVALUASI TARI DENOK      ")
    print("="*40)
    print(f"1. Skor Urutan (Sequence) : {result['score_sequence']:.2f} / 100")
    print(f"2. Skor Postur (Teknik)   : {result['score_posture']:.2f} / 100")
    print("-" * 40)
    print(f"NILAI AKHIR               : {result['final_score']:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()