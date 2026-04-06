#I am archith and this CV code is 40% AI and 60% self coded .Any doubts ping me on my website
#https://archith-portfolio-omega.vercel.app/

import cv2
import numpy as np
import tensorflow as tf
import time

# ──────── LOAD MODEL ────────
MODEL_PATH = "traffic_sign_model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded!")

CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for heavy vehicles',
    11: 'Right-of-way at intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Heavy vehicles prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left',
    20: 'Dangerous curve right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all limits', 33: 'Turn Right',
    34: 'Turn Left', 35: 'Ahead only', 36: 'Right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by heavy vehicles'
}

# ──────── DISTANCE & POSITION ────────
KNOWN_SIGN_WIDTH_CM = 60
FOCAL_LENGTH = 500

def estimate_distance(sign_width_pixels):
    if sign_width_pixels <= 0:
        return 0
    return (KNOWN_SIGN_WIDTH_CM * FOCAL_LENGTH) / sign_width_pixels

def get_sign_position(x1, x2, frame_width):
    center_x = (x1 + x2) // 2
    third = frame_width // 3
    if center_x < third:
        return "LEFT"
    elif center_x > 2 * third:
        return "RIGHT"
    else:
        return "CENTER"

# ──────── PREDICT ────────
def predict(roi):
    img = cv2.resize(roi, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    class_id = np.argmax(predictions)
    confidence = predictions[class_id]
    return CLASS_NAMES[class_id], confidence

# ──────── DETECT SIGNS ────────
def detect_signs(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_regions = []

    red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([135, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([10, 50, 50]), np.array([40, 255, 255]))

    combined_mask = red_mask | blue_mask | yellow_mask
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 800 or area > 150000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w / h < 0.4 or w / h > 2.5:
            continue
        pad = 15
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        detected_regions.append((x1, y1, x2, y2))

    return detected_regions

# ──────── MAIN LOOP (UPDATED FOR DROIDCAM) ────────
print("Looking for DroidCam...")

cap = None
# DroidCam usually mounts to index 1, but we check 0 and 2 just in case
for cam_id in [0, 1, 2]:
    test_cap = cv2.VideoCapture(cam_id)
    if test_cap.isOpened():
        # DroidCam REQUIREMENT: Set resolution before reading frames to prevent green screen
        test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        ret, frame = test_cap.read()
        if ret and frame is not None:
            cap = test_cap
            print(f"✅ DroidCam connected securely at index {cam_id}")
            break
        else:
            test_cap.release()

if cap is None:
    print("❌ No camera found! Check if the DroidCam PC client is open and streaming video.")
    exit()

CONFIDENCE_THRESHOLD = 0.50

fps_start = time.time()
fps_count = 0
fps_display = 0

print("🎥 Camera started. Show a traffic sign. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    # SAFETY NET: If DroidCam lags and sends an empty/green frame, skip it instead of crashing
    if not ret or frame is None:
        print("⚠️ Waiting for video data from phone...")
        cv2.waitKey(100)
        continue

    frame_h, frame_w = frame.shape[:2]

    # Detect signs
    regions = detect_signs(frame)

    for (x1, y1, x2, y2) in regions:
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] < 20 or roi.shape[1] < 20:
            continue

        sign_name, confidence = predict(roi)

        if confidence > CONFIDENCE_THRESHOLD:
            distance = estimate_distance(x2 - x1)
            position = get_sign_position(x1, x2, frame_w)

            # Green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Sign name (green label)
            label = f"{sign_name} ({confidence*100:.0f}%)"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + label_size[0] + 5, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Distance + Position (yellow, below box)
            cv2.putText(frame, f"Dist: {distance:.0f}cm | {position}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # FPS counter
    fps_count += 1
    if time.time() - fps_start >= 1.0:
        fps_display = fps_count
        fps_count = 0
        fps_start = time.time()

    cv2.putText(frame, f"FPS: {fps_display}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "'q' = quit", (10, frame_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Traffic Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Camera stopped.")