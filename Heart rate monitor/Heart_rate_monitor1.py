import cv2
import numpy as np
import time
import tensorflow as tf
from tkinter import Tk, Label, Frame, Button, Canvas, Toplevel
from PIL import Image, ImageTk
from cvzone.FaceDetectionModule import FaceDetector
from ttkbootstrap import Style, Button as ttkButton
from tkinter.ttk import Progressbar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Configuration ===
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_WIDTH = 160
ROI_HEIGHT = 120
CHANNELS = 3
FPS = 15

# === Load Emotion Detection Model ===
emotion_model = tf.keras.models.load_model("emotion_model_mini_XCEPTION.hdf5", compile=False)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# === Webcam and Face Detection Setup ===
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
detector = FaceDetector()

# === Signal Processing ===
levels = 3
alpha = 170
minFreq = 1.0
maxFreq = 2.0
bufferSize = 150
bufferIndex = 0

bpmCalcInterval = 10
bpmHistorySize = 10
bpmHistory = np.zeros(bpmHistorySize)
bpmIndex = 0

firstFrame = np.zeros((ROI_HEIGHT, ROI_WIDTH, CHANNELS), dtype=np.float32)
gaussRef = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(firstFrame)))
gaussBuffer = np.zeros((bufferSize, *gaussRef.shape), dtype=np.float32)
fftAvg = np.zeros(bufferSize)

frequencies = FPS * np.arange(bufferSize) / bufferSize
freqMask = (frequencies >= minFreq) & (frequencies <= maxFreq)

# === GUI Setup with ttkbootstrap ===
style = Style("darkly")
root = style.master
root.title("💓 Heart & Emotion Monitor")
root.geometry("1000x800")
root.configure(bg="#101820")

header_frame = Frame(root, bg="#101820")
header_frame.pack(pady=5)

Label(header_frame, text=" Heart Rate & Emotion Detection ", font=("Segoe UI", 22, "bold"), bg="#101820", fg="#00ffe1").pack()

video_label = Label(root, bg="#101820")
video_label.pack(pady=10)

info_frame = Frame(root, bg="#101820")
info_frame.pack(pady=10)

bpm_label = Label(info_frame, text="BPM: --", font=("Segoe UI", 18, "bold"), bg="#101820", fg="#00ff8c")
bpm_label.grid(row=0, column=0, padx=40)

emotion_label = Label(info_frame, text="Emotion: --", font=("Segoe UI", 18, "bold"), bg="#101820", fg="#ffc107")
emotion_label.grid(row=0, column=1, padx=40)

# Emojis for emotions
emoji_map = {
    "Happiness": "😊",
    "Sadness": "😢",
    "Anger": "😠",
    "Fear": "😨",
    "Surprise": "😲",
    "Disgust": "🤢",
    "Neutral": "😐"
}

canvas_width = 700
canvas_height = 200
ecg_canvas = Canvas(root, width=canvas_width, height=canvas_height, bg="#000000", bd=0, highlightthickness=0)
ecg_canvas.pack(pady=15)
ecg_data = [canvas_height // 2] * canvas_width

# Add FPS label and BPM progress bar
fps_label = Label(info_frame, text="FPS: --", font=("Segoe UI", 14), bg="#101820", fg="#00ffe1")
fps_label.grid(row=1, column=0, columnspan=2, pady=5)

bpm_bar = Progressbar(root, orient="horizontal", length=400, mode="determinate", maximum=150)
bpm_bar.pack(pady=10)

# Add Exit Button with ttkbootstrap style
ttkButton(root, text="Exit", bootstyle="danger", command=root.destroy).pack(pady=10)

prevTime = time.time()
frameCount = 0

# === Helper Functions ===
def build_gaussian_pyramid(frame, levels):
    pyramid = [frame.astype(np.float32)]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame.astype(np.float32))
    return pyramid

def reconstruct_from_pyramid(pyramid, index, levels):
    frame = pyramid[index]
    for _ in range(levels):
        frame = cv2.pyrUp(frame)
    return frame[:ROI_HEIGHT, :ROI_WIDTH]

def generate_ecg_sample(bpm, t):
    if bpm is None or bpm <= 0:
        bpm = 75
    cycle = 60 / bpm
    pos = t % cycle
    height = canvas_height // 2

    if pos < 0.05 * cycle:
        return height - 10
    elif pos < 0.06 * cycle:
        return height + 30
    elif pos < 0.065 * cycle:
        return height - 50
    elif pos < 0.07 * cycle:
        return height + 30
    elif pos < 0.12 * cycle:
        return height - 5
    else:
        return height + np.random.randint(-2, 3)

def draw_3d_ecg(data):
    ecg_canvas.delete("wave")
    for i in range(1, len(data)):
        glow_width = 4 + int(np.sin(i * 0.1) * 1.5)
        ecg_canvas.create_line(i - 1, data[i - 1] + 2, i, data[i] + 2, fill="#003300", width=glow_width, tags="wave")
        ecg_canvas.create_line(i - 1, data[i - 1], i, data[i], fill="#00ff8c", width=2, tags="wave")

# === Update GUI ===
def update_gui():
    global bufferIndex, frameCount, bpmIndex, prevTime, ecg_data

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    frame, faces = detector.findFaces(frame, draw=False)
    display = frame.copy()

    currentTime = time.time()
    fps = 1 / (currentTime - prevTime + 1e-6)
    prevTime = currentTime

    bpmVal = '--'
    emotion = '--'
    emoji = ""  # Default emoji value

    if faces:
        x, y, w, h = faces[0]['bbox']
        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (ROI_WIDTH, ROI_HEIGHT))

        pyramid = build_gaussian_pyramid(roi, levels + 1)
        if pyramid[levels].shape == gaussBuffer[bufferIndex].shape:
            gaussBuffer[bufferIndex] = pyramid[levels]

            fftData = np.fft.fft(gaussBuffer, axis=0)
            fftData[~freqMask] = 0
            frameCount += 1

            if frameCount % bpmCalcInterval == 0:
                for i in range(bufferSize):
                    fftAvg[i] = np.real(fftData[i]).mean()
                hz = frequencies[np.argmax(fftAvg)]
                bpm = 60.0 * hz
                bpm = max(bpm, 1)
                bpmHistory[bpmIndex] = bpm
                bpmIndex = (bpmIndex + 1) % bpmHistorySize

            filtered = np.real(np.fft.ifft(fftData, axis=0)) * alpha
            enhanced = cv2.convertScaleAbs(roi + reconstruct_from_pyramid(filtered, bufferIndex, levels))
            bufferIndex = (bufferIndex + 1) % bufferSize

            bpmVal = int(bpmHistory.mean())

            # Emotion detection
            face_img = cv2.resize(roi, (64, 64))  # Resize to model expected input
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            face_img = face_img.astype("float32") / 255.0
            face_img = np.expand_dims(face_img, axis=-1)  # (64, 64, 1)
            face_img = np.expand_dims(face_img, axis=0)   # (1, 64, 64, 1)

            emotion_pred = emotion_model.predict(face_img, verbose=0)
            emotion_idx = np.argmax(emotion_pred)
            emotion = emotion_labels[emotion_idx]
            
            

            # Get emoji for emotion
            emoji = emoji_map.get(emotion, "")

            t = time.time()
            beat = generate_ecg_sample(bpmVal if isinstance(bpmVal, int) and bpmVal > 0 else 75, t)
            ecg_data.append(beat)
            if len(ecg_data) > canvas_width:
                ecg_data = ecg_data[-canvas_width:]
            draw_3d_ecg(ecg_data)

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    bpm_label.config(text=f"BPM: {bpmVal}")
    emotion_label.config(text=f"Emotion: {emotion} {emoji}")
    fps_label.config(text=f"FPS: {int(fps)}")
    bpm_bar['value'] = bpmVal

    # Convert OpenCV image (BGR) to Tkinter image format (RGB)
    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(display_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.img_tk = img_tk
    video_label.config(image=img_tk)

    root.after(10, update_gui)

update_gui()
root.mainloop()
