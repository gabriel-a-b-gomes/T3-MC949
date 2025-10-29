import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import threading
from queue import Queue
import pyttsx3
from gtts import gTTS
from playsound import playsound
import tempfile

############################################
# CONFIG PRINCIPAL
############################################
FAST_MODE = True
FRAME_SKIP = 2
USE_FLOW = False

############################################
# 1) MODELOS: YOLO + MiDaS
############################################
yolo = YOLO('yolov8n.pt')
device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')

midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True).to(device).eval()
transform = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True).small_transform

############################################
# 2) CONFIGS DO PROJETO
############################################
CLASSES_UTEIS = ['person','bicycle','car','bench','chair','stop sign','potted plant','truck','motorcycle','bus','traffic light', 'fire hydrant', 'parking meter', 'dog', 'cat', 'chair', 'couch', 'bed' ]

def get_roi(frame):
    h, w = frame.shape[:2]
    x1, x2 = int(w*0.10), int(w*0.90)
    y1, y2 = int(h*0.35), int(h*0.95)
    return (x1, y1, x2, y2)

DIST_DANGER_M = 2.5
FLOW_APPROACH_MIN = 0.6
CENTER_BONUS = 0.15

############################################
# 3) FUNÇÕES UTILITÁRIAS (com calibração automática + debug visual)
############################################

FOV_H_DEG = 75.0
PERSON_HEIGHT_M = 1.55
CAMERA_HEIGHT_M = 1.0
_dynamic_scale = 1.0
_focal_px = None
_last_debug_dist = None
_calibrating_frames = 0

def predict_depth(frame_bgr):
    if FAST_MODE:
        frame_bgr = cv2.resize(frame_bgr, (frame_bgr.shape[1]//2, frame_bgr.shape[0]//2))
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
    pred_norm = cv2.medianBlur((pred_norm*255).astype(np.uint8), 5)
    pred_norm = pred_norm.astype(np.float32)/255.0
    if FAST_MODE:
        pred_norm = cv2.resize(pred_norm, (frame_bgr.shape[1]*2, frame_bgr.shape[0]*2))
    return pred_norm

def _auto_calibrate_scale(yolo_results, invdepth, frame_shape):
    """
    Ajusta automaticamente o fator de escala com base em uma pessoa detectada.
    """
    global _dynamic_scale, _focal_px, _last_debug_dist, _calibrating_frames
    h, w = frame_shape[:2]
    if _focal_px is None:
        _focal_px = (w / 2) / np.tan(np.deg2rad(FOV_H_DEG / 2))

    person_boxes = []
    for r in yolo_results:
        for b in r.boxes:
            cls = yolo.names[int(b.cls)]
            if cls == 'person' and float(b.conf) > 0.6:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cx = (x1 + x2) / 2
                person_boxes.append((abs(cx - w/2), (x1, y1, x2, y2)))
    if not person_boxes:
        return

    _, (x1, y1, x2, y2) = min(person_boxes, key=lambda t: t[0])
    h_box = max(1, y2 - y1)

    d_geom = (_focal_px * PERSON_HEIGHT_M) / h_box

    box_depth = invdepth[y1:y2, x1:x2]
    if box_depth.size == 0:
        return
    inv_med = float(np.median(box_depth))
    d_midas = _dynamic_scale / max(inv_med, 1e-3)

    scale_factor = d_geom / max(d_midas, 1e-3)
    _dynamic_scale = 0.9 * _dynamic_scale + 0.1 * (_dynamic_scale * scale_factor)

    _last_debug_dist = d_geom
    _calibrating_frames += 1

def invdepth_to_meters(invdepth):
    inv = max(float(invdepth), 1e-3)
    d_midas = _dynamic_scale / inv
    dh = max(0.0, PERSON_HEIGHT_M - CAMERA_HEIGHT_M)
    if d_midas > dh:
        d_midas = (d_midas ** 2 - dh ** 2) ** 0.5
    return d_midas

def bbox_center(b):
    x1, y1, x2, y2 = b
    return (int((x1 + x2)//2), int((y1 + y2)//2))

def mean_flow_in_bbox(flow, bbox):
    if not USE_FLOW or flow is None:
        return 0.0, 0.0, 0.0
    x1, y1, x2, y2 = bbox
    fx, fy = flow[y1:y2, x1:x2, 0], flow[y1:y2, x1:x2, 1]
    if fx.size == 0:
        return 0.0, 0.0, 0.0
    mag = np.sqrt(fx*fx + fy*fy)
    return float(np.mean(fx)), float(np.mean(fy)), float(np.mean(mag))

def center_weight_in_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    rx, ry = (x1 + x2)/2, (y1 + y2)/2
    rw, rh = (x2 - x1), (y2 - y1)
    dx, dy = abs(cx - rx)/(rw/2 + 1e-6), abs(cy - ry)/(rh/2 + 1e-6)
    d = np.sqrt(dx*dx + dy*dy)
    return max(0.0, 1.0 - min(1.0, d))

############################################
# 4) TTS + RATE-LIMITING
############################################

def _init_tts_engine():
    # eng = pyttsx3.init()
    # try:
    #     for v in eng.getProperty('voices'):
    #         name = (v.name or '').lower()
    #         lang = ''.join(v.languages).lower() if hasattr(v, 'languages') else ''
    #         if 'pt' in lang or 'portugu' in name:
    #             eng.setProperty('voice', v.id)
    #             if 'br' in lang or 'braz' in name:
    #                 break
    # except: pass
    # eng.setProperty('rate', 180)
    # eng.setProperty('volume', 1.0)
    # return eng
    return


_tts_engine = _init_tts_engine()
_tts_queue = Queue(maxsize=6)
_tts_lock = threading.Lock()

def _tts_worker():
    last_phrase = None
    while True:
        t = _tts_queue.get()
        if last_phrase != t:
            try:
                with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as f:
                    tts = gTTS(t, lang='pt-br')
                    tts.save(f.name)
                    playsound(f.name)
            except: pass
            last_phrase = t
        _tts_queue.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

_last_speak_ts = 0.0
_last_alert_ts = 0.0
MIN_SPEAK_GAP_S = 0.1
MIN_ALERT_GAP_S = 0.1
per_target_cooldown = {}
TARGET_COOLDOWN_S = 0.5

NOMES_PT = {
    'person': 'pessoa',
    'bicycle': 'bicicleta',
    'car': 'carro',
    'bench': 'banco',
    'chair': 'cadeira',
    'stop sign': 'placa de pare',
    'potted plant': 'planta',
    'truck': 'caminhão',
    'motorcycle': 'moto',
    'bus': 'ônibus',
    'traffic light': 'semáforo',
    'fire hydrant': 'hidrante',
    'parking meter': 'parcómetro',
    'dog': 'cão',
    'cat': 'gato',
    'chair': 'cadeira',
    'couch': 'sofá',
    'bed': 'cama',
}

def _fmt_metros(x): return f"{x:.1f}".replace('.', ',')
def _dir_from_bbox(center, roi):
    cx, cy = center; x1, y1, x2, y2 = roi
    t = (x1 + (x2 - x1)/2, x1 + (x2 - x1)/2)
    return "à esquerda" if cx < t[0] else "à direita" if cx > t[1] else "à frente"

def speak(txt, priority=False, frame_idx):
    global _last_speak_ts, _last_alert_ts
    snow = time.time()
    if priority:
        if now - _last_alert_ts < MIN_ALERT_GAP_S: return
        with _tts_lock:
            try:
                while not _tts_queue.empty():
                    _tts_queue.get_nowait(); _tts_queue.task_done()
            except: pass
        _last_alert_ts = now
    else:
        if now - _last_speak_ts < MIN_SPEAK_GAP_S: return
        _last_speak_ts = now
    try:
        _tts_queue.put_nowait(txt)
    except: pass

############################################
# 5) LOOP PRINCIPAL
############################################

video_path = 'praca_edat.mp4'
cap = cv2.VideoCapture(video_path)
prev_gray = None
frame_idx = 0
last_direction = None
obstaculos = {}

while True:
    ok, frame = cap.read()
    if not ok: break
    frame_idx += 1
    if FAST_MODE and frame_idx % (FRAME_SKIP+1) != 0:
        continue

    start = time.time()

    h, w = frame.shape[:2]
    roi = get_roi(frame)
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 2)

    invdepth = predict_depth(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = None
    if USE_FLOW:
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        prev_gray = gray.copy()

    results = yolo(frame, verbose=False)
    _auto_calibrate_scale(results, invdepth, frame.shape)

    candidates = []
    for r in results:
        for b in r.boxes:
            cls = yolo.names[int(b.cls)]; conf = float(b.conf)
            if conf < 0.45 or cls not in CLASSES_UTEIS: continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cx, cy = bbox_center((x1, y1, x2, y2))
            if not (roi[0] < cx < roi[2] and roi[1] < cy < roi[3]): continue

            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            box_depth = invdepth[y1c:y2c, x1c:x2c]
            if box_depth.size == 0: continue
            inv_med = float(np.median(box_depth))
            dist_m = invdepth_to_meters(inv_med)
            mx, my, mmag = mean_flow_in_bbox(flow, (x1c, y1c, x2c, y2c))
            approach = mmag + max(0.0, -my)
            cweight = center_weight_in_roi(cx, cy, roi)
            candidates.append({
                'bbox': (x1, y1, x2, y2),
                'cls': cls,
                'conf': conf,
                'dist_m': dist_m,
                'flow_mag': mmag,
                'approach': approach,
                'center_w': cweight,
                'center': (cx, cy)
            })

    target = None
    if candidates:
        for c in candidates:
            inv_d = 1.0 / (c['dist_m'] + 1e-6)
            c['score'] = 1.5*inv_d + 1.0*c['center_w'] + 0.8*c['approach']
        target = max(candidates, key=lambda x: x['score'])

    for c in candidates:
        x1, y1, x2, y2 = c['bbox']
        color = (0, 0, 255) if target and c is target else (0, 255, 0)
        txt = f"{c['cls']} {c['dist_m']:.1f}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, txt, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- DEBUG VISUAL ---
    status = "CALIBRANDO..." if _calibrating_frames < 15 else "OK"
    scale_txt = f"Escala: {_dynamic_scale:.2f} | {status}"
    cv2.putText(frame, scale_txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
    if _last_debug_dist:
        cv2.putText(frame, f"Ref pessoa: {_last_debug_dist:.1f}m", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    # --------------------

    alert_text = ""
    if target:
        near = target['dist_m'] <= DIST_DANGER_M
        fast = target['approach'] >= FLOW_APPROACH_MIN
        centered = target['center_w'] >= (1.0 - CENTER_BONUS)
        if near:#(near and fast) or (near and centered):
            alert_text = f"ALERTA: {target['cls']} a {target['dist_m']:.1f} m"
            cv2.putText(frame, alert_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

            obstaculos[target['bbox']] = frame_idx

            #print(alert_text)

    for obstaculo in obstaculos.copy():
        if frame_idx - obstaculos[obstaculo] > 90:
            obstaculos.pop(obstaculo)
        else:
            x1, y1, x2, y2 = obstaculo
            color = (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # if target:
    #     direction = _dir_from_bbox(target['center'], roi)
    #     alvo_key = (target['cls'], direction)
    #     if alert_text:
    #         cls_pt = NOMES_PT.get(target['cls'], target['cls'])
    #         dist_txt = _fmt_metros(target['dist_m'])
    #         acao = "Pare." if direction == "à frente" else \
    #                "Desvie à esquerda." if direction == "à direita" else "Desvie à direita."
    #         fala = f"Alerta! {cls_pt} a {dist_txt} metros {direction}. {acao}"
    #         speak(fala, priority=True)
    #         print("*" + fala)
    #     else:
    #         now = time.time()
    #         if now - per_target_cooldown.get(alvo_key, 0.0) >= TARGET_COOLDOWN_S:
    #             cls_pt = NOMES_PT.get(target['cls'], target['cls'])
    #             dist_txt = _fmt_metros(target['dist_m'])
    #             speak(f"{cls_pt} a {dist_txt} metros {direction}.", priority=False)
    #             per_target_cooldown[alvo_key] = now
    #             print(f"{cls_pt} a {dist_txt} metros {direction}.")
    if alert_text:
        direction = _dir_from_bbox(target['center'], roi)
        if (direction != last_direction):
            cls_pt = NOMES_PT.get(target['cls'], target['cls'])
            fala = f"{cls_pt} {direction}"
            speak(fala, priority=True)
            last_direction = direction
    elif last_direction is not None:
        fala = "Caminho livre"
        speak(fala)
        last_direction = None

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow("Etapa 4", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
