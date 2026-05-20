import cv2
import numpy as np
import pytesseract
import re
import threading
import time
import pickle
import os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ==========================================
# 1. KALIBRASI ROI DENGAN KLIK (hanya sekali)
# ==========================================
CONFIG_FILE = "roi_config.pkl"
rtmp_url = "rtmp://192.168.100.29:1935/live"

# Variabel kalibrasi
pts = []
calibrated = False
frame_to_show = None

def click_handler(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts = [(x, y)] if len(pts) >= 2 else pts + [(x, y)]
        print(f"Titik kiri-atas: {pts[0]}" if len(pts) == 1 else f"Titik kanan-bawah: {pts[1]}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(pts) == 1:
            pts.append((x, y))
            print(f"Titik kanan-bawah: {pts[1]}")

def load_or_calibrate():
    global roi_x, roi_y, roi_w, roi_h, calibrated
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "rb") as f:
            roi = pickle.load(f)
            roi_x, roi_y, roi_w, roi_h = roi
            print(f"ROI dimuat dari file: {roi}")
            return True

    print("\n=== KALIBRASI AREA GPS ===")
    print("1. Klik kiri di pojok kiri-atas teks GPS")
    print("2. Klik kanan di pojok kanan-bawah teks GPS")
    print("3. Tekan 'y' untuk menyimpan dan melanjutkan\n")

    cap_temp = cv2.VideoCapture(rtmp_url)
    if not cap_temp.isOpened():
        print("Tidak bisa membuka stream untuk kalibrasi.")
        return False

    ret, frame = cap_temp.read()
    cap_temp.release()
    if not ret:
        print("Gagal mengambil frame kalibrasi.")
        return False

    global frame_to_show
    frame_to_show = frame.copy()
    cv2.namedWindow("Kalibrasi - Klik kiri & kanan, lalu 'y'")
    cv2.setMouseCallback("Kalibrasi - Klik kiri & kanan, lalu 'y'", click_handler)

    while True:
        disp = frame_to_show.copy()
        for pt in pts:
            cv2.circle(disp, pt, 5, (0,255,0), -1)
        if len(pts) == 2:
            cv2.rectangle(disp, pts[0], pts[1], (0,255,0), 2)
        cv2.imshow("Kalibrasi - Klik kiri & kanan, lalu 'y'", disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('y') and len(pts) == 2:
            roi_x, roi_y = pts[0]
            roi_w = pts[1][0] - pts[0][0]
            roi_h = pts[1][1] - pts[0][1]
            with open(CONFIG_FILE, "wb") as f:
                pickle.dump((roi_x, roi_y, roi_w, roi_h), f)
            print(f"ROI disimpan: {roi_x},{roi_y},{roi_w},{roi_h}")
            cv2.destroyWindow("Kalibrasi - Klik kiri & kanan, lalu 'y'")
            return True
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()
    return False

calibrated = load_or_calibrate()
if not calibrated:
    print("Kalibrasi gagal. Keluar.")
    exit()

# ==========================================
# 2. MULTITHREAD OCR + TAMPILAN
# ==========================================
latest_frame = None
frame_lock = threading.Lock()
latest_roi = None
roi_lock = threading.Lock()
gps_text = "Menunggu GPS..."
stop_event = threading.Event()

# Thread pembaca video (tanpa menumpuk)
def video_reader():
    global latest_frame
    cap = cv2.VideoCapture(rtmp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue
        with frame_lock:
            latest_frame = frame.copy()
    cap.release()

# Thread OCR
def ocr_worker():
    global gps_text
    while not stop_event.is_set():
        with roi_lock:
            roi = latest_roi.copy() if latest_roi is not None else None
        if roi is None:
            time.sleep(0.2)
            continue
        try:
            # Preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            # Cari koordinat
            match = re.search(r'(-?\d+\.\d+)\s*[,;]\s*(-?\d+\.\d+)', text)
            if match:
                lat, lon = match.group(1), match.group(2)
                gps_text = f"Lat: {lat}, Lon: {lon}"
                print(f"[GPS] {gps_text}")
        except:
            pass
        time.sleep(0.6)

# Jalankan thread
reader = threading.Thread(target=video_reader, daemon=True)
ocr = threading.Thread(target=ocr_worker, daemon=True)
reader.start()
ocr.start()

print("Sistem berjalan. Tekan 'q' untuk keluar.")

frame_count = 0
while not stop_event.is_set():
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        time.sleep(0.01)
        continue

    # Update ROI untuk OCR
    frame_count += 1
    if frame_count % 15 == 0:
        h, w = frame.shape[:2]
        x1, y1 = max(0, roi_x), max(0, roi_y)
        x2, y2 = min(w, roi_x+roi_w), min(h, roi_y+roi_h)
        if x2 > x1 and y2 > y1:
            with roi_lock:
                latest_roi = frame[y1:y2, x1:x2]

    # Overlay
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0,255,255), 2)
    cv2.putText(frame, gps_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Drone GPS Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

cv2.destroyAllWindows()