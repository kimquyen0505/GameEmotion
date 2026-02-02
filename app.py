from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
import time
import random
import os
from collections import deque, Counter

# --- IMPORT MEDIAPIPE (CÁCH AN TOÀN CHO MAC) ---
try:
    import mediapipe as mp
    # Sử dụng cách gọi trực tiếp qua solutions
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    print(f"Lỗi khởi tạo MediaPipe: {e}")
    MEDIAPIPE_AVAILABLE = False

app = Flask(__name__)

# --- 🧠 THUẬT TOÁN AI NHẬN DIỆN CHUẨN (TỈ LỆ HÌNH HỌC) ---
class EmotionDetector:
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True, # Lấy chi tiết môi
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        self.history = deque(maxlen=4)

    def predict(self, lms, w, h):
        def get_pt(idx): return np.array([lms.landmark[idx].x * w, lms.landmark[idx].y * h])
        
        # Thước đo chuẩn: Khoảng cách giữa 2 mắt (Inter-ocular distance)
        # Điểm 33 và 263 là khóe mắt ngoài
        base_dist = np.linalg.norm(get_pt(33) - get_pt(263))
        if base_dist == 0: return "neutral"

        # 1. Tỉ lệ há miệng dọc (Mouth Open) - Điểm 13, 14
        m_h = np.linalg.norm(get_pt(13) - get_pt(14)) / base_dist
        
        # 2. Tỉ lệ rộng miệng ngang (Mouth Width) - Điểm 61, 291
        m_w = np.linalg.norm(get_pt(61) - get_pt(291)) / base_dist
        
        # 3. Độ mở mắt (Eye Open) - Điểm 159, 145 và 386, 374
        eye_l = np.linalg.norm(get_pt(159) - get_pt(145)) / base_dist
        eye_r = np.linalg.norm(get_pt(386) - get_pt(374)) / base_dist
        avg_eye = (eye_l + eye_r) / 2

        # 4. Độ nhếch khóe môi (Smile score)
        # So sánh cao độ y của khóe môi so với môi trên (điểm 0)
        smile_lift = (get_pt(0)[1] - (get_pt(61)[1] + get_pt(291)[1]) / 2) / base_dist

        # 5. Độ nhướng lông mày
        brow_lift = np.linalg.norm(get_pt(105) - get_pt(159)) / base_dist

        # --- LOGIC QUYẾT ĐỊNH ---
        res = "neutral"

        # BẤT NGỜ: Mắt mở to (>0.25) + Lông mày nhướng (>0.2) + Miệng há dọc
        if avg_eye > 0.26 and brow_lift > 0.22:
            res = "surprise"
        
        # LÈ LƯỠI: Khi lưỡi thò ra, miệng há dọc cực đại (>0.45)
        elif m_h > 0.48:
            res = "tongue_out"
            
        # VUI VẺ: Khóe môi nhếch lên cao (>0.06) hoặc miệng mở rất rộng ngang
        elif smile_lift > 0.07 or m_w > 0.95:
            res = "happy"

        self.history.append(res)
        return Counter(self.history).most_common(1)[0][0]

class RhythmGame:
    def __init__(self):
        self.detector = EmotionDetector()
        self.score = 0
        self.combo = 0
        self.bpm = 80 # Tăng tốc độ game một chút
        self.start_time = time.time()
        self.last_beat_int = -1
        self.curr_target = "neutral"
        self.next_target = random.choice(["happy", "surprise", "tongue_out"])
        self.player_emo = "neutral"
        self.feedback = ""
        self.feedback_time = 0
        self.is_hit = False

    def process(self, frame):
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        
        elapsed = time.time() - self.start_time
        beat_dur = 60 / self.bpm
        beat_prog = (elapsed / beat_dur) % 4
        curr_beat_int = int(elapsed / beat_dur)

        # Đổi nhịp (Beat Logic)
        if curr_beat_int > self.last_beat_int:
            if curr_beat_int % 4 == 0:
                self.curr_target = self.next_target
                self.next_target = random.choice(["happy", "surprise", "tongue_out"])
                self.is_hit = False
            self.last_beat_int = curr_beat_int

        # AI Detect
        if MEDIAPIPE_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0]
                self.player_emo = self.detector.predict(lms, w, h)
                
                # Vẽ điểm môi và mắt để người chơi biết AI đang hoạt động
                for idx in [13, 14, 61, 291, 159, 386]:
                    p = lms.landmark[idx]
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 2, (0, 255, 0), -1)

        # Hit Logic (Kiểm tra xem người chơi làm đúng biểu cảm ở nhịp 4 không)
        dist_to_beat = min(beat_prog, abs(4 - beat_prog))
        if dist_to_beat < 0.5 and not self.is_hit:
            if self.player_emo == self.curr_target and self.curr_target != "neutral":
                self.score += 10 + self.combo
                self.combo += 1
                self.feedback = "PERFECT!"
                self.feedback_time = time.time()
                self.is_hit = True

        # HUD đơn giản trên camera
        cv2.rectangle(frame, (50, h-30), (int(50 + (beat_prog/4)*(w-100)), h-10), (0, 255, 0), -1)
        return frame

game = RhythmGame()

# --- ROUTES ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/assets/<path:filename>')
def assets(filename): return send_from_directory('assets', filename)

@app.route('/get_emotion')
def get_emotion(): return jsonify({"emotion": game.player_emo})

@app.route('/game_data')
def game_data():
    return jsonify({
        "score": game.score,
        "combo": game.combo,
        "target": game.curr_target,
        "next": game.next_target,
        "player_emo": game.player_emo,
        "feedback": game.feedback if time.time() - game.feedback_time < 0.8 else ""
    })

def gen_frames():
    # Thử ID 1 trước cho máy bạn, nếu không được tự về 0
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success: break
        frame = game.process(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(port=5001, debug=False)