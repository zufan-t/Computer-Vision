import cv2
import time
import os
from modules.pose_engine import PoseEngine
from modules.scoring import ScoreCalculator


def main():

    path_guru = "data/Denok_master.mp4"
    path_siswa = "data/Denok_murid.mp4"

    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "hasil_analisis_sitari.mp4")

    cap_guru = cv2.VideoCapture(path_guru)
    cap_siswa = cv2.VideoCapture(path_siswa)

    detector_guru = PoseEngine(complexity=1)
    detector_siswa = PoseEngine(complexity=1)

    scorer = ScoreCalculator()

    w_guru = int(cap_guru.get(3))
    h_guru = int(cap_guru.get(4))

    target_h = 480
    target_w = int((w_guru / h_guru) * target_h)

    final_w = target_w * 2
    final_h = target_h

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (final_w, final_h))

    print(f"Mulai Analisis... Hasil akan disimpan di {save_path}")

    list_nilai_postur = []
    list_nilai_timing = []

    nilai_postur = 0
    nilai_timing = 0
    warna_timing = (0, 0, 255)

    frame_count = 0
    while True:
        frame_count += 1
        if frame_count % 500 == 0:
            print(f"Processing frame {frame_count}...")

        success_g, img_guru = cap_guru.read()
        success_s, img_siswa = cap_siswa.read()

        if not success_g:
            cap_guru.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success_g, img_guru = cap_guru.read()

        if not success_s:
            break



        img_guru = cv2.resize(img_guru, (target_w, target_h))
        img_siswa = cv2.resize(img_siswa, (target_w, target_h))

        if frame_count % 5 == 0:

            img_guru, lm_guru = detector_guru.find_pose(img_guru, draw=False)

            img_siswa, lm_siswa = detector_siswa.find_pose(img_siswa, draw=True)

            nilai_postur = 0
            nilai_timing = 0

            if len(lm_guru) != 0 and len(lm_siswa) != 0:
                nilai_postur, nilai_timing = scorer.calculate_score(lm_guru, lm_siswa)

                list_nilai_postur.append(nilai_postur)
                list_nilai_timing.append(nilai_timing)
        else:

             pass

        img_final = cv2.hconcat([img_guru, img_siswa])

        cv2.rectangle(img_final, (10, 10), (250, 120), (0, 0, 0), cv2.FILLED)
        cv2.putText(img_final, f"POSTUR (Wiraga)", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        cv2.putText(img_final, f"{int(nilai_postur)}%", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)

        cv2.putText(img_final, f"TIMING (Wirama)", (20, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

        warna_timing = (0, 255, 0) if nilai_timing > 80 else (0, 0, 255)
        cv2.putText(img_final, f"{int(nilai_timing)}%", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 3, warna_timing, 3)

        out.write(img_final)

    cap_guru.release()
    cap_siswa.release()
    out.release()
    cv2.destroyAllWindows()

    avg_postur = sum(list_nilai_postur) / len(list_nilai_postur) if list_nilai_postur else 0
    avg_timing = sum(list_nilai_timing) / len(list_nilai_timing) if list_nilai_timing else 0

    final_score = (avg_postur + avg_timing) / 2

    report = ("\n" + "="*40 + "\n" +
              "      RAPOR EVALUASI TARI DENOK      \n" +
              "="*40 + "\n" +
              f"1. Skor Postur (Wiraga)   : {avg_postur:.2f} / 100\n" +
              f"2. Skor Timing (Wirama)   : {avg_timing:.2f} / 100\n" +
              "-" * 40 + "\n" +
              f"NILAI AKHIR               : {final_score:.2f}\n" +
              "="*40 + "\n")

    print(report)
    with open(os.path.join(output_dir, "rapor.txt"), "w") as f:
        f.write(report)

    print("Selesai! Video hasil analisis tersimpan di folder output.")

if __name__ == "__main__":
    main()