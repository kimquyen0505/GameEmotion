from flask import Flask, render_template, Response, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque, Counter

app = Flask(__name__)

lock = threading.Lock()
outputFrame = None
current_emotion = "neutral"
system_state = "init" 

class GeometryEmotionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.history = deque(maxlen=4)
        
        self.calib_frames = []
        self.baseline = None 
    
    def get_face_stats(self, lms, w, h):
        def get_pt(idx): return np.array([lms.landmark[idx].x * w, lms.landmark[idx].y * h])
        def dist(i1, i2): return np.linalg.norm(get_pt(i1) - get_pt(i2))

        # Chiều rộng khuôn mặt (làm chuẩn)
        face_width = dist(234, 454)
        if face_width == 0: return None

        return {
            # 1. Các chỉ số cho ANGRY
            "brow_dist": dist(107, 336) / face_width,         # Khoảng cách 2 đầu lông mày
            "brow_height": ((dist(65, 159) + dist(295, 386)) / 2) / face_width, # Độ cao lông mày
            
            # 2. Các chỉ số cho HAPPY
            "mouth_width": dist(61, 291) / face_width,        # Độ rộng miệng (Cười bè)
            "smile_curve": (get_pt(0)[1] - (get_pt(61)[1] + get_pt(291)[1])/2) / face_width, # Độ cong môi (Cười tươi)
            
            # 3. Chỉ số cho SURPRISE
            "mouth_height": dist(13, 14) / face_width,        # Độ mở dọc
        }

    def predict(self, frame):
        global system_state
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        emotion = "neutral"
        debug_info = {}

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0]
            curr = self.get_face_stats(lms, w, h)
            if not curr: return "neutral", {}

            # --- GIAI ĐOẠN 1: HỌC (Calibration) ---
            if system_state == "calibrating":
                self.calib_frames.append(curr)
                progress = len(self.calib_frames)
                
                if progress >= 30: # Học trong 30 khung hình
                    avg = {}
                    for k in curr:
                        avg[k] = sum(x[k] for x in self.calib_frames) / 30
                    self.baseline = avg
                    system_state = "running"
                    print("--> ĐÃ HỌC XONG CẤU TRÚC MẶT!")
                
                return "neutral", {"mode": "calib", "prog": int(progress/30*100)}

            # --- GIAI ĐOẠN 2: CHẠY (Running) ---
            elif system_state == "running" and self.baseline:
                base = self.baseline
                
                # --- TÍNH TOÁN TỶ LỆ (%) THAY ĐỔI ---
                # r > 1.0: Lớn hơn bình thường | r < 1.0: Nhỏ hơn bình thường
                
                r_mouth_w = curr["mouth_width"] / base["mouth_width"]     # Độ giãn miệng
                r_mouth_h = curr["mouth_height"] / base["mouth_height"]   # Độ há miệng
                r_frown   = curr["brow_dist"] / base["brow_dist"]         # Độ cau mày
                r_brow_h  = curr["brow_height"] / base["brow_height"]     # Độ nhướng mày
                
                # --- LOGIC NHẬN DIỆN MỚI ---
                
                # 1. KIỂM TRA HAPPY (VUI VẺ) - Cập nhật mới
                # Logic: Cười là khi miệng RỘNG HƠN (bè sang ngang) HOẶC môi cong lên
                is_wide_smile = r_mouth_w > 1.10  # Rộng hơn 10% so với lúc nghiêm túc
                is_curved_smile = curr["smile_curve"] > 0.025 # Độ cong cổ điển
                
                # Điều kiện loại trừ: Nếu há miệng quá to theo chiều dọc (la hét) thì ko phải cười
                not_screaming = r_mouth_h < 1.6
                
                if (is_wide_smile or is_curved_smile) and not_screaming:
                    emotion = "happy"
                
                # 2. KIỂM TRA SURPRISE (BẤT NGỜ)
                # Há mồm to gấp đôi HOẶC nhướng mày rất cao
                elif (r_mouth_h > 1.8) or (r_brow_h > 1.15):
                    emotion = "surprise"
                
                # 3. KIỂM TRA ANGRY (TỨC GIẬN) - Logic giữ nguyên vì đã ổn
                # Không cười và (Cau mày chặt HOẶC Hạ thấp lông mày)
                elif not (is_wide_smile or is_curved_smile):
                    # Co lông mày < 95% HOẶC (Hạ lông mày < 92% VÀ Nhíu mày nhẹ)
                    if (r_frown < 0.95) or (r_brow_h < 0.93):
                        emotion = "angry"

                # DEBUG info vẽ lên màn hình
                debug_info = {
                    "mode": "run",
                    "smile%": int(r_mouth_w * 100),   # > 110% -> Happy
                    "curve":  curr["smile_curve"],    # > 0.025 -> Happy
                    "frown%": int(r_frown * 100)      # < 95% -> Angry
                }

        self.history.append(emotion)
        # Tăng buffer lên 4 để đỡ nhấp nháy
        final_emo = Counter(list(self.history)[-4:]).most_common(1)[0][0]
        return final_emo, debug_info

def camera_loop():
    global outputFrame, current_emotion, system_state
    
    # Reset về trạng thái học mỗi khi bật
    system_state = "calibrating" 
    
    detector = GeometryEmotionDetector()
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            emo, vals = detector.predict(frame)
            
            if vals.get("mode") == "calib":
                # MÀN HÌNH ĐANG HỌC
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                msg = f"LOADING... {vals['prog']}%"
                # Hướng dẫn màu vàng
                cv2.putText(frame, "KHONG CUOI - DE MAT NGHIEM", (w//2-250, h//2-40), 1, 2, (0,255,255), 3)
                cv2.putText(frame, msg, (w//2-140, h//2+50), 1, 1.8, (255,255,255), 2)
                
            elif vals.get("mode") == "run":
                # HIỂN THỊ CHỈ SỐ HAPPY ĐỂ BẠN TEST
                # Nếu Width > 110% màu tím (Happy)
                s_score = vals["smile%"]
                col_s = (200,200,200)
                if s_score > 110 or vals["curve"] > 0.025: col_s = (0,255,0) # Xanh lá
                
                # Hiển thị độ giãn miệng (Test Happy)
                cv2.putText(frame, f"Gian Mieng: {s_score}% (>110)", (20, 40), 1, 1.2, col_s, 2)
                
                # Hiển thị độ cau mày (Test Angry)
                f_score = vals["frown%"]
                col_f = (0,0,255) if f_score < 95 else (200,200,200)
                cv2.putText(frame, f"Cau May: {f_score}% (<95)", (20, 80), 1, 1.2, col_f, 2)
                
                # Hiển thị tên cảm xúc
                colors = {"happy":(0,255,0), "angry":(0,0,255), "surprise":(255,0,255), "neutral":(255,255,0)}
                cv2.putText(frame, f"--> {emo.upper()}", (20, 160), 1, 2.5, colors.get(emo,(255,255,255)), 4)

            with lock: outputFrame = frame.copy(); current_emotion = emo
        else: time.sleep(0.1)

t = threading.Thread(target=camera_loop, daemon=True).start()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/assets/<path:filename>')
def assets(filename): return send_from_directory('assets', filename)

def generate():
    while True:
        with lock:
            if outputFrame is None: time.sleep(0.01); continue
            (flag, enc) = cv2.imencode(".jpg", outputFrame)
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(enc) + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed(): return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion(): return jsonify({"emotion": current_emotion, "state": system_state})

if __name__ == "__main__":
    app.run(port=5001, debug=False, threaded=True)