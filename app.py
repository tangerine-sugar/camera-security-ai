from flask import Flask, render_template, Response, jsonify # <--- Nhớ thêm jsonify
import cv2
import os
import numpy as np
import time

app = Flask(__name__)

# --- CẤU HÌNH ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
is_trained = False
last_alert_time = 0

# Biến toàn cục để lưu trạng thái hiện tại (SAFE hoặc DANGER)
current_status = "safe" 

# --- HÀM HUẤN LUYỆN  ---
def train_model():
    global is_trained
    faces = []
    ids = []
    path = 'known_faces'
    if not os.path.exists(path): os.makedirs(path)
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_rect = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces_rect:
                faces.append(gray[y:y+h, x:x+w])
                ids.append(1)
        except: pass
    
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        is_trained = True
        print(">>> MODEL TRAINED!")

train_model()
camera = cv2.VideoCapture(0)

# --- XỬ LÝ VIDEO ---
def generate_frames():
    global last_alert_time, current_status # <--- Gọi biến toàn cục
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        # Mặc định là an toàn nếu không có ai
        if len(faces) == 0:
            current_status = "safe"

        # Khai báo biến đếm thời gian (để bên ngoài vòng lặp generate_frames hoặc dùng global)
        
        for (x, y, w, h) in faces:
            # --- MÃ MÀU ANSI ---
            RED = "\033[91m"
            GREEN = "\033[92m"
            BOLD = "\033[1m"
            RESET = "\033[0m"
            # -------------------

            if is_trained:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                
                # --- TRƯỜNG HỢP 1: NGƯỜI QUEN (ADMIN) ---
                if confidence < 80: 
                    name = "ADMIN"
                    color = (0, 255, 0) # Khung XANH
                    current_status = "safe"
                    
                    # (Tùy chọn) In 1 dòng nhẹ nhàng báo Admin đang đăng nhập
                    # print(f"{GREEN} >> Admin Detected. (Safe){RESET}", end='\r') 

                # --- TRƯỜNG HỢP 2: NGƯỜI LẠ (INTRUDER) ---
                else:
                    name = "INTRUDER"
                    current_status = "danger"
                    color = (0, 0, 255) # Khung ĐỎ

                    # --- LOGIC BÁO ĐỘNG 1 LẦN (COOLDOWN) ---
                    current_time = time.time()
                    
                    # Chỉ in nếu đã qua 5 giây kể từ lần báo trước
                    if (current_time - last_alert_time) > 5:
                        print("\n" + "="*50)
                        print(f"{RED}{BOLD} 🚨 CẢNH BÁO: PHÁT HIỆN NGƯỜI LẠ! {RESET}")
                        print(f"{RED}    >>> Mức độ sai lệch: {round(confidence)}")
                        print("="*50 + "\n")
                        
                        last_alert_time = current_time # Cập nhật lại thời gian để chờ tiếp
            
            # Phần vẽ khung hình (để khung luôn hiện màu đỏ thời gian thực)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- CÁC ROUTE WEB ---
@app.route('/')
def intro(): return render_template('intro.html')

@app.route('/monitor')
def monitor(): return render_template('monitor.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API ĐỂ WEB HỎI TÌNH TRẠNG ---
@app.route('/status')
def get_status():
    return jsonify({'status': current_status})

if __name__ == '__main__':
    app.run(debug=True)