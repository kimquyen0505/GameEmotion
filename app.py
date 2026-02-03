from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp # Import chuẩn
import numpy as np
import threading
import time
from collections import deque, Counter

app = Flask(__name__)

# --- GLOBAL VARIABLES ---
outputFrame = None
current_emotion = "neutral"
lock = threading.Lock()

# --- AI ENGINE ---
class EmotionDetector:
    def __init__(self):
        # MediaPipe Solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.history = deque(maxlen=5)

    def predict(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        emotion = "neutral"
        
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0]
            
            def dist(i1, i2):
                p1 = np.array([lms.landmark[i1].x, lms.landmark[i1].y])
                p2 = np.array([lms.landmark[i2].x, lms.landmark[i2].y])
                return np.linalg.norm(p1 - p2)

            # Chuẩn hóa theo chiều rộng mặt
            face_width = dist(33, 263)
            if face_width == 0: return "neutral"

            mouth_open = dist(13, 14) / face_width
            mouth_wide = dist(61, 291) / face_width
            brow_lift = dist(65, 159) / face_width
            
            # --- LOGIC NHẬN DIỆN ---
            # 1. Happy: Miệng rộng > 0.55
            if mouth_wide > 0.55:
                emotion = "happy"
            
            # 2. Surprise: Miệng mở > 0.15 VÀ Lông mày cao > 0.19
            elif mouth_open > 0.15 and brow_lift > 0.19:
                emotion = "surprise"
                
            # 3. Tongue Out: Miệng mở > 0.15 NHƯNG Lông mày thấp (<= 0.19)
            elif mouth_open > 0.15 and brow_lift <= 0.19:
                emotion = "tongue_out"
                
        self.history.append(emotion)
        final_emotion = Counter(self.history).most_common(1)[0][0]
        return final_emotion

# --- CAMERA LOOP ---
def camera_loop():
    global outputFrame, current_emotion
    # Try to initialize the detector; if MediaPipe isn't available, disable camera loop
    try:
        detector = EmotionDetector()
    except Exception as e:
        print("Camera loop disabled: could not initialize EmotionDetector:", e)
        return
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            # Nhận diện
            emo = detector.predict(frame)
            
            # Vẽ chữ DEBUG lên hình để bạn kiểm tra
            cv2.putText(frame, f"AI: {emo}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            with lock:
                outputFrame = frame.copy()
                current_emotion = emo
        else:
            time.sleep(0.1)

t = threading.Thread(target=camera_loop, daemon=True)
t.start()

# --- WEB SERVER ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/assets/<path:filename>')
def assets(filename):
    from flask import send_from_directory
    # Serve generated audio from the project's `assets/` folder
    # (previously pointed to 'static/sounds' which doesn't contain the generated files)
    return send_from_directory('assets', filename)

def generate():
    while True:
        with lock:
            if outputFrame is None:
                time.sleep(0.01)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion(): return jsonify({"emotion": current_emotion})

if __name__ == "__main__":
    # Run without the debug reloader to avoid interference with background
    # camera threads and to provide a stable single-process server.
    app.run(port=5001, debug=False, threaded=True)