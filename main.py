# Drishti Mithra - Assistive vision app (voice-controlled object/text detection)
# For simplicity, all modules (speech, detection, OCR, control) are implemented here.
# In a production version, these can be modularized into separate scripts.

import platform
import shutil
import time
import json
import threading

import cv2
import numpy as np
import pyttsx3
import pytesseract
import requests
import sounddevice as sd
import vosk
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# PHONE CONFIG - using the "IP Webcam" Android app for both camera and GPS.
# Open the app on your phone, tap "Start server", and it will show an
# address like http://192.168.1.5:8080 - put that exact address below
# (no trailing slash).
# ---------------------------------------------------------------------------
IP_WEBCAM_BASE_URL = "http://192.168.110.57:8080"  # <-- CHANGE THIS to your phone's address
IP_WEBCAM_VIDEO_URL = f"{IP_WEBCAM_BASE_URL}/video"
IP_WEBCAM_SENSORS_URL = f"{IP_WEBCAM_BASE_URL}/sensors.json"

# ---------------------------------------------------------------------------
# FIX #6: Cross-platform Tesseract path
# Instead of hardcoding the Windows path, try to auto-detect tesseract on
# the current OS, and only fall back to the Windows path if we're on Windows.
# ---------------------------------------------------------------------------
def configure_tesseract():
    found = shutil.which("tesseract")
    if found:
        pytesseract.pytesseract.tesseract_cmd = found
    elif platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # else: leave default; will raise a clear TesseractNotFoundError if missing


configure_tesseract()

# ---------------------------------------------------------------------------
# Text-to-speech
# Confirmed on this machine: a single reused pyttsx3 engine produces no
# audio, but a fresh engine.init() per call, run on its own thread (the
# original approach), works reliably. So we keep that exact pattern rather
# than the queued single-engine version.
# ---------------------------------------------------------------------------
def speak(text):
    def speak_thread():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=speak_thread, daemon=True).start()


# ---------------------------------------------------------------------------
# FIX #5: Voice recognition
# The Vosk model is large and slow to load. Loading it fresh on every call to
# listen() adds noticeable lag each time a command is captured. Load it once
# and reuse it.
# ---------------------------------------------------------------------------
_vosk_model_cache = {}


def get_vosk_model(model_path):
    if model_path not in _vosk_model_cache:
        _vosk_model_cache[model_path] = vosk.Model(model_path)
    return _vosk_model_cache[model_path]


def listen(model_path='vosk-model-small-en-us-0.15', duration=4):
    model = get_vosk_model(model_path)
    rec = vosk.KaldiRecognizer(model, 16000)

    def callback(indata, frames, time_, status):
        rec.AcceptWaveform(bytes(indata))

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                            channels=1, callback=callback):
        sd.sleep(int(duration * 1000))

    res = json.loads(rec.FinalResult())
    return res.get('text', '').lower().strip()


# ---------------------------------------------------------------------------
# FIX #7: Allow voice "stop" to interrupt the camera loop.
# Previously, once capture_and_detect() started, the only way out was
# pressing 'q' on the video window - the voice command "stop" was never
# heard because listen() wasn't called again until capture_and_detect
# returned. We now run a small background listener thread that watches for
# "stop"/"exit" while the camera loop runs, and sets a flag to break out.
# ---------------------------------------------------------------------------
def _background_stop_listener(stop_event, vosk_model_path):
    while not stop_event.is_set():
        command = listen(vosk_model_path, duration=2)
        if 'stop' in command or 'exit' in command:
            stop_event.set()
            break


# ---------------------------------------------------------------------------
# GPS via IP Webcam's /sensors.json endpoint.
# NOTE: the exact JSON layout can vary slightly by app version. This parser
# tries the common shape IP Webcam uses, but if it can't find GPS data it
# prints the raw response once so you can see the real field names on your
# phone's version and we can adjust the parsing in one place.
# ---------------------------------------------------------------------------
_gps_format_warned = False


def get_gps(timeout=2):
    global _gps_format_warned
    try:
        resp = requests.get(IP_WEBCAM_SENSORS_URL, timeout=timeout)
        data = resp.json()
    except Exception as e:
        print(f"GPS request failed: {e}")
        return None

    try:
        gps_entry = data.get("gps")
        if gps_entry and gps_entry.get("data"):
            latest = gps_entry["data"][-1]      # [timestamp_ms, [lat, lon, accuracy_or_alt]]
            lat, lon = latest[1][0], latest[1][1]
            return {"lat": lat, "lon": lon}
    except Exception:
        pass

    if not _gps_format_warned:
        print("Could not parse GPS from sensors.json - here's the raw response so we can fix the parser:")
        print(json.dumps(data, indent=2)[:2000])
        _gps_format_warned = True
    return None


def capture_and_detect(model, vosk_model_path='vosk-model-small-en-us-0.15'):
    cap = cv2.VideoCapture(IP_WEBCAM_VIDEO_URL)
    last_spoken = {}
    SPEAK_COOLDOWN = 5

    if not cap.isOpened():
        speak("Camera not detected.")
        return

    speak("Camera started.")

    stop_event = threading.Event()
    listener_thread = threading.Thread(
        target=_background_stop_listener,
        args=(stop_event, vosk_model_path),
        daemon=True
    )
    listener_thread.start()

    while True:
        if stop_event.is_set():
            speak("Stopping camera.")
            break

        ret, frame = cap.read()

        if not ret:
            speak("Unable to read camera.")
            break

        detections = detect_objects(frame, model)
        text = read_text(frame)

        if text:
            print("OCR:", text)

        guidance = generate_guidance(detections)
        current_time = time.time()

        # FIX #2: Removed the duplicate speak block that was repeated twice
        # in a row inside this loop (dead code - the second copy could never
        # actually fire since last_spoken[message] was just updated).
        for message in guidance:
            print("AI:", message)
            last_time = last_spoken.get(message, 0)
            if current_time - last_time >= SPEAK_COOLDOWN:
                speak(message)
                last_spoken[message] = current_time

        # Draw detections on the camera screen
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            label = f"{d['label']} - {d['position']}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Drishti Mithra", frame)

        # FIX #3: Single, correctly-placed quit check (was duplicated/
        # misplaced inside the guidance loop before, where it could be
        # skipped entirely on frames with no detections).
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


def detect_objects(frame, model):
    results = model.predict(source=frame, conf=0.3, verbose=False)

    detections = []
    height, width, _ = frame.shape

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = r.names[cls]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            box_width = x2 - x1

            center_x = (x1 + x2) // 2

            if center_x < width // 3:
                position = "left"
            elif center_x > (2 * width) // 3:
                position = "right"
            else:
                position = "center"

            distance = estimate_distance(box_width)

            detections.append({
                'label': label,
                'conf': conf,
                'bbox': [x1, y1, x2, y2],
                'position': position,
                'distance': distance
            })

    return detections


def estimate_distance(box_width):
    if box_width <= 0:
        return None
    distance = 500 / box_width
    return round(distance, 1)


def read_text(frame):
    text = pytesseract.image_to_string(frame)
    return text.strip()


def generate_guidance(detections):
    guidance = []

    for d in detections:
        label = d['label']
        position = d['position']
        distance = d['distance']

        # FIX #1: distance can be None (e.g. a zero-width bounding box).
        # The original code did `if distance <= 0.8:` directly, which
        # crashes with a TypeError when distance is None. We now guard
        # against that and fall back to a message with no distance.
        if distance is None:
            if position == "left":
                message = f"{label} on your left."
            elif position == "right":
                message = f"{label} on your right."
            else:
                message = f"{label} ahead."
        elif distance <= 0.8:
            if position == "left":
                message = f"Warning! {label} very close on your left. Stop."
            elif position == "right":
                message = f"Warning! {label} very close on your right. Stop."
            else:
                message = f"Warning! {label} very close ahead. Stop."
        else:
            if position == "left":
                message = f"{label} approximately {distance} meters on your left."
            elif position == "right":
                message = f"{label} approximately {distance} meters on your right."
            else:
                message = f"{label} approximately {distance} meters ahead."

        guidance.append(message)

    return guidance


def describe(detections):
    if not detections:
        return "No objects detected."
    labels = [d['label'] for d in detections]
    return "Detected objects are: " + ", ".join(labels)


def main():
    model = YOLO('yolov8n.pt')
    vosk_model_path = 'vosk-model-small-en-us-0.15'
    speak("Shivam Aayush and Vedant are ready. Say capture to start or stop to exit.")

    # Quick one-time GPS check at startup, so you can confirm the phone
    # connection works before relying on it for navigation later.
    location = get_gps()
    if location:
        print(f"GPS OK: lat={location['lat']}, lon={location['lon']}")
    else:
        print("GPS not available yet (check phone connection / IP_WEBCAM_BASE_URL).")

    while True:
        command = listen(vosk_model_path)
        print("Command:", command)
        if 'capture' in command:
            capture_and_detect(model, vosk_model_path)
        elif 'stop' in command or 'exit' in command or 'top' in command:
            speak("Thank you for using us. Goodbye!")
            break
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()