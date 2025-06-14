import cv2
import numpy as np
import time
import sys
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import tensorflow as tf  # AI/ML library (TensorFlow)
from sklearn.preprocessing import StandardScaler

# === Configuration ===
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_WIDTH = 160
ROI_HEIGHT = 120
CHANNELS = 3
FPS = 15

# === Load Pretrained Model ===
# Load a pre-trained model (e.g., for emotion detection or stress prediction)
emotion_model = tf.keras.models.load_model("emotion_detection_model.h5")

# === Webcam Initialization ===
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
detector = FaceDetector()

# === Signal Processing Config ===
levels = 3
alpha = 170
minFreq = 1.0
maxFreq = 2.0
bufferSize = 150  # Buffer size for signal processing
bufferIndex = 0

# === BPM Calculation Config ===
bpmCalcInterval = 10
bpmHistorySize = 10
bpmHistory = np.zeros(bpmHistorySize)
bpmIndex = 0
plotY = LivePlot(FRAME_WIDTH, FRAME_HEIGHT, [60, 120], invert=True)

# === Feature Extraction and Preprocessing for AI ===
scaler = StandardScaler()  # Feature scaling

# === Helper Functions ===
def build_gaussian_pyramid(frame, levels):
    pyramid = [frame]
    for _ in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstruct_from_pyramid(pyramid, index, levels):
    frame = pyramid[index]
    for _ in range(levels):
        frame = cv2.pyrUp(frame)
    return frame[:ROI_HEIGHT, :ROI_WIDTH]

# === Buffers Initialization ===
firstFrame = np.zeros((ROI_HEIGHT, ROI_WIDTH, CHANNELS), dtype=np.float32)
gaussRef = build_gaussian_pyramid(firstFrame, levels + 1)[levels]
gaussBuffer = np.zeros((bufferSize, *gaussRef.shape), dtype=np.float32)
fftAvg = np.zeros(bufferSize)

frequencies = (1.0 * FPS) * np.arange(bufferSize) / bufferSize
freqMask = (frequencies >= minFreq) & (frequencies <= maxFreq)

# === Main Loop ===
prevTime = time.time()
frameCount = 0

print("[INFO] Starting Heart Rate Monitor. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    frame, faces = detector.findFaces(frame, draw=False)
    display = frame.copy()

    # FPS calculation
    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime
    cv2.putText(display, f'FPS: {int(fps)}', (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if faces:
        x, y, w, h = faces[0]['bbox']
        cv2.rectangle(display, (x, y), (x+w, y+h), (255, 0, 255), 2)

        roi = frame[y:y+h, x:x+w]
        roi = cv2.resize(roi, (ROI_WIDTH, ROI_HEIGHT))
        gaussBuffer[bufferIndex] = build_gaussian_pyramid(roi, levels+1)[levels]

        fftData = np.fft.fft(gaussBuffer, axis=0)
        fftData[~freqMask] = 0

        frameCount += 1  # Increment every frame

        if frameCount % bpmCalcInterval == 0:
            for i in range(bufferSize):
                fftAvg[i] = np.real(fftData[i]).mean()

            hz = frequencies[np.argmax(fftAvg)]
            bpm = 60.0 * hz
            bpmHistory[bpmIndex] = bpm
            bpmIndex = (bpmIndex + 1) % bpmHistorySize

        filtered = np.real(np.fft.ifft(fftData, axis=0)) * alpha
        enhanced = cv2.convertScaleAbs(roi + reconstruct_from_pyramid(filtered, bufferIndex, levels))

        bufferIndex = (bufferIndex + 1) % bufferSize

        display[0:ROI_HEIGHT//2, FRAME_WIDTH-ROI_WIDTH//2:] = cv2.resize(enhanced, (ROI_WIDTH//2, ROI_HEIGHT//2))

        bpmVal = bpmHistory.mean()
        plot = plotY.update(float(bpmVal))

        # Emotion Detection via AI (optional)
        # Extract face ROI and preprocess for emotion prediction
        face_roi = cv2.resize(roi, (48, 48))  # Resize to match model input
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_roi = face_roi / 255.0  # Normalize
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Predict emotion (this can help in understanding stress or other factors affecting heart rate)
        emotion_pred = emotion_model.predict(face_roi)
        emotion = np.argmax(emotion_pred)

        # Display emotion on the screen
        emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
        cv2.putText(display, f'Emotion: {emotion_labels[emotion]}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if frameCount > bpmHistorySize:
            cvzone.putTextRect(display, f'BPM: {int(bpmVal)}', (FRAME_WIDTH//2 - 80, 40), scale=2, thickness=2, colorR=(0, 200, 100))
        else:
            cvzone.putTextRect(display, "Analyzing...", (30, 40), scale=2, thickness=2)

        stacked = cvzone.stackImages([display, plot], cols=2, scale=1)
    else:
        cvzone.putTextRect(display, "Face not detected", (30, 40), scale=2, thickness=2, colorR=(0, 0, 255))
        stacked = cvzone.stackImages([display, display], cols=2, scale=1)

    cv2.imshow("Heart Rate Monitor", stacked)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
