import csv
import cv2

def salva_fps(fps_list, time_list):
    with open("fps_log.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["tempo_s", "fps"])
        for t, f_val in zip(time_list, fps_list):
            writer.writerow([f"{t:.3f}".replace('.', ','), f"{f_val:.2f}".replace('.', ',')])
    print("FPS salvos em fps_log.csv")

import csv

def salva_latencia_csv(latency_log):
    with open("latency_log.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["tempo_s", "texto", "latencia_s"])
        for ts, texto, lat in latency_log:
            # converte float para string com vírgula
            writer.writerow([f"{ts:.3f}".replace('.', ','), texto, f"{lat:.3f}".replace('.', ',')])
    print("Latência salva em latency_log.csv")

def armazena_alerta(alert_frames_dir, frame_idx, frame, alert_text, alert_log):
    filename = f"{alert_frames_dir}/frame_{frame_idx:06d}.jpg"
    cv2.imwrite(filename, frame)
    alert_log.append((frame_idx, alert_text))

def salva_alertas(alert_log):
    with open("alert_log.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["frame_idx", "alerta"])
        for frame_idx, texto in alert_log:
            writer.writerow([frame_idx, texto])
    print("Log de alertas salvo em alert_log.csv")

def salva_estabilidade_class(class_stability_log):
    with open("class_stability.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["frame_idx", "classe", "mudou"])
        for frame_idx, cls, changed in class_stability_log:
            writer.writerow([frame_idx, cls, changed])
    print("Estabilidade de classe salva em class_stability.csv")