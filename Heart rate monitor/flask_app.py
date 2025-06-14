from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import tensorflow as tf
from cvzone.FaceDetectionModule import FaceDetector

app = Flask(__name__)

# === Webcam Setup ===
cap = cv2.VideoCapture(0)

# === Load Emotion Detection Model ===
emotion_model = tf.keras.models.load_model("emotion_model_mini_XCEPTION.hdf5", compile=False)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# === Face Detector ===
detector = FaceDetector()

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, faces = detector.findFaces(frame, draw=False)
        bpmVal = "--"
        emotion = "--"

        if faces:
            x, y, w, h = faces[0]['bbox']
            roi = frame[y:y+h, x:x+w]
            roi = cv2.resize(roi, (64, 64))
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            face_img = gray.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)

            emotion_pred = emotion_model.predict(face_img, verbose=0)
            emotion_idx = np.argmax(emotion_pred)
            emotion = emotion_labels[emotion_idx]

            bpmVal = 70 + int(10 * np.sin(time.time()))  # Simulated BPM

            cv2.putText(frame, f"BPM: {bpmVal}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
