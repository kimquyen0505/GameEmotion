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

class EmotionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.history = deque(maxlen=4)

    def predict(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        emotion = "neutral"
        debug_vals = {}

        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0]
            
            def get_pt(idx): return np.array([lms.landmark[idx].x * w, lms.landmark[idx].y * h])
            def dist(i1, i2): return np.linalg.norm(get_pt(i1) - get_pt(i2))

            # --- 1. LẤY CÁC THÔNG SỐ CỐT LÕI ---
            
            # Thước đo chuẩn: Độ rộng khuôn mặt (từ thái dương này sang kia)
            # Dùng cái này ổn định hơn khoảng cách mắt
            face_width = dist(234, 454) 
            if face_width == 0: return "neutral", {}

            # Độ mở miệng DỌC (Cao)
            m_height = dist(13, 14)
            # Độ mở miệng NGANG (Rộng)
            m_width = dist(61, 291)
            
            # TỶ LỆ KHUNG MIỆNG (QUAN TRỌNG NHẤT)
            # Ratio < 0.4: Miệng dẹt (Cười hoặc Bình thường)
            # Ratio > 0.5: Miệng tròn/dọc (Bất ngờ hoặc Lè lưỡi)
            mouth_ratio = m_height / m_width

            # Độ nhướng mày (Chuẩn hóa theo độ rộng mặt)
            # Mắt (159) lên Lông mày (65)
            brow_dist = dist(65, 159)
            brow_ratio = brow_dist / face_width # > 0.08 là nhướng

            # Độ nhếch mép (Smile Curve)
            # Mép môi (61) có cao hơn môi giữa (0) không? (Trục Y ngược)
            lip_center_y = get_pt(0)[1]
            corner_y = (get_pt(61)[1] + get_pt(291)[1]) / 2
            smile_score = (lip_center_y - corner_y) / face_width # > 0.02 là cười

            # --- 2. LOGIC PHÂN LOẠI "KHẮC TINH" ---

            # A. KIỂM TRA HAPPY (VUI VẺ) TRƯỚC
            # Điều kiện:
            # 1. Khóe miệng phải nhếch lên (Smile Score > 0.02)
            # 2. HOẶC Miệng bè ngang rất rộng (m_width lớn) VÀ Ratio thấp (không há hốc mồm)
            if (smile_score > 0.025) or (m_width/face_width > 0.45 and mouth_ratio < 0.6):
                emotion = "happy"

            # B. KIỂM TRA SURPRISE (BẤT NGỜ)
            # Điều kiện:
            # 1. Miệng phải mở DỌC (Ratio > 0.5) HOẶC mở rất to
            # 2. VÀ Lông mày phải nhướng cao (Brow > 0.085)
            elif (mouth_ratio > 0.4 or m_height/face_width > 0.15) and (brow_ratio > 0.085):
                emotion = "surprise"

            # C. KIỂM TRA TONGUE OUT (LÈ LƯỠI)
            # Điều kiện:
            # 1. Miệng mở DỌC (Ratio > 0.4)
            # 2. NHƯNG Lông mày thấp (<= 0.085) - Đây là điểm khác biệt với Surprise
            elif (mouth_ratio > 0.4 or m_height/face_width > 0.15) and (brow_ratio <= 0.085):
                emotion = "tongue_out"
            
            # D. CÒN LẠI
            else:
                emotion = "neutral"

            # --- DEBUG INFO ---
            debug_vals = {
                "SMILE": f"{smile_score:.3f}",  # Cười: > 0.025
                "BROW": f"{brow_ratio:.3f}",   # Nhướng: > 0.085
                "SHAPE": f"{mouth_ratio:.2f}"   # <0.5: Dẹt, >0.5: Tròn
            }

        self.history.append(emotion)
        final_emo = Counter(self.history).most_common(1)[0][0]
        return final_emo, debug_vals

def camera_loop():
    global outputFrame, current_emotion
    try: detector = EmotionDetector()
    except: return
    cap = cv2.VideoCapture(1)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            emo, vals = detector.predict(frame)
            
            # --- VẼ HƯỚNG DẪN TRỰC TIẾP ---
            if vals:
                # 1. SMILE SCORE (Cười)
                s_val = float(vals["SMILE"])
                col_s = (0,255,0) if s_val > 0.025 else (0,0,255)
                cv2.putText(frame, f"SMILE: {vals['SMILE']} (>0.025)", (20, 50), 1, 1.2, col_s, 2)
                
                # 2. BROW (Lông mày - Phân biệt Surprise/Lè lưỡi)
                b_val = float(vals["BROW"])
                col_b = (0,255,255) # Vàng = Lè lưỡi (Thấp)
                if b_val > 0.085: col_b = (255,0,255) # Tím = Surprise (Cao)
                cv2.putText(frame, f"BROW:  {vals['BROW']} (>0.085)", (20, 90), 1, 1.2, col_b, 2)
                
                # 3. MOUTH SHAPE (Hình dáng miệng)
                m_val = float(vals["SHAPE"])
                shape_txt = "DET (Cười)" if m_val < 0.5 else "TRON (Ha)"
                cv2.putText(frame, f"SHAPE: {m_val} ({shape_txt})", (20, 130), 1, 1.2, (200,200,200), 2)

            # KẾT QUẢ
            cv2.putText(frame, f"--> {emo.upper()}", (20, 180), 1, 2, (0,255,0), 3)

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
def get_emotion(): return jsonify({"emotion": current_emotion})

if __name__ == "__main__":
    app.run(port=5001, debug=False, threaded=True)