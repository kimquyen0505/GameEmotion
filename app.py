from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
import threading
import time
import os
import wave
import struct
import math
from collections import deque, Counter

# Kiểm tra MediaPipe
try:
    import mediapipe as mp
    from mediapipe.python.solutions import face_mesh
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError, ModuleNotFoundError):
    MEDIAPIPE_AVAILABLE = False

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
global_frame = None
current_emotion = "neutral"
lock = threading.Lock()

# --- HÀM TẠO ÂM THANH (ĐẢM BẢO KHÔNG LỖI ASSETS) ---
def generate_melody(filename, notes):
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    path = os.path.join(assets_dir, filename)
    if os.path.exists(path): return
    sample_rate = 22050
    audio_data = b''
    for freq, duration in notes:
        num_samples = int(duration * sample_rate)
        for i in range(num_samples):
            sample = math.sin(2 * math.pi * freq * i / sample_rate)
            audio_data += struct.pack('<h', int(sample * 32767))
    with wave.open(path, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sample_rate); f.writeframes(audio_data)

sounds_to_gen = {
    'beat.wav': [(440, 0.1)], 'success.wav': [(523, 0.1), (784, 0.2)],
    'fail.wav': [(392, 0.2), (261, 0.3)], 'v_prompt.wav': [(659, 0.1), (784, 0.2)],
    'v_correct.wav': [(1047, 0.1), (1319, 0.2)], 'v_wrong.wav': [(220, 0.3)],
    'v_gameover.wav': [(440, 0.2), (261, 0.5)]
}
for name, notes in sounds_to_gen.items():
    generate_melody(name, notes)

# --- 🧠 AI NHẬN DIỆN CẢI TIẾN ĐỘ CHÍNH XÁC ---
class EmotionDetector:
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = face_mesh.FaceMesh(
                max_num_faces=1, 
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            self.use_mediapipe = True
        else:
            self.use_mediapipe = False
        self.history = deque(maxlen=6) # Làm mượt cảm xúc để không bị nháy

    def predict(self, frame):
        if not self.use_mediapipe: return "neutral"
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        
        emotion = "neutral"
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0]
            def pt(i): return np.array([lms.landmark[i].x * w, lms.landmark[i].y * h])
            def dist(p1, p2): return np.linalg.norm(pt(p1) - pt(p2))

            # CHUẨN HÓA DỌC (Từ trán điểm 10 đến cằm điểm 152) - Ổn định hơn chiều ngang
            face_h = dist(10, 152)
            if face_h < 10: return "neutral"

            # 1. MIỆNG
            m_wide = dist(61, 291) / face_h       # Rộng miệng
            m_open_v = dist(13, 14) / face_h      # Há miệng dọc
            # Độ nhếch khóe môi (Khóe môi y so với môi dưới y)
            corner_y = (pt(61)[1] + pt(291)[1]) / 2
            smile_lift = (pt(14)[1] - corner_y) / face_h

            # 2. MẮT & LÔNG MÀY (Cho biểu cảm bất ngờ)
            eye_open = (dist(159, 145) + dist(386, 374)) / (2 * face_h)
            brow_h = (dist(105, 159) + dist(336, 386)) / (2 * face_h)

            # LOGIC QUYẾT ĐỊNH
            if m_wide > 0.45 and smile_lift > 0.04:
                emotion = "happy"
            elif eye_open > 0.10 and brow_h > 0.20:
                emotion = "surprise"
            elif m_open_v > 0.18:
                emotion = "tongue_out"

        self.history.append(emotion)
        return Counter(self.history).most_common(1)[0][0]

detector = EmotionDetector()

# --- 📸 XỬ LÝ CAMERA (SỬA LỖI MÀN HÌNH ĐEN TRÊN MAC) ---
def camera_stream():
    global global_frame, current_emotion
    
    # Mac Pro đôi khi nhận Camera rời/Continuity là index 1, tích hợp là 0
    cap = None
    for idx in [0, 1]:
        temp_cap = cv2.VideoCapture(1)
        if temp_cap.isOpened():
            ret, _ = temp_cap.read()
            if ret:
                cap = temp_cap
                print(f"✅ Đã nhận diện Camera tại Index: {1}")
                break
        temp_cap.release()

    if cap is None:
        print("❌ LỖI: Không tìm thấy Camera. Hãy kiểm tra quyền truy cập hệ thống.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1) # Lật gương
            emo = detector.predict(frame)
            with lock:
                global_frame = frame.copy()
                current_emotion = emo
        else:
            time.sleep(0.1)

threading.Thread(target=camera_stream, daemon=True).start()

# --- FLASK ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if global_frame is None: continue
                _, buffer = cv2.imencode('.jpg', global_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion(): return jsonify({"emotion": current_emotion})

@app.route('/assets/<path:filename>')
def assets(filename): return send_from_directory('assets', filename)

if __name__ == "__main__":
    # TẮT debug=True để tránh việc Flask nạp lại 2 lần gây khóa Camera trên Mac
    app.run(port=5001, debug=False, threaded=True)