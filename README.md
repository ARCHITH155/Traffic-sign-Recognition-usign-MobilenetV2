# 🚦 Real-Time Traffic Sign Detection & Recognition

A real-time traffic sign detection and recognition system built with **TensorFlow Lite** and **OpenCV**. It uses a MobileNetV2-based model to classify **43 types of traffic signs** from the [GTSRB dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html), with live distance estimation and positional tracking.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Real-Time Detection** | HSV color-space filtering (red, blue, yellow) with contour analysis to locate signs in each frame |
| **43-Class Recognition** | Classifies all GTSRB sign categories — speed limits, prohibitions, warnings, and mandatory signs |
| **Distance Estimation** | Approximates how far the sign is from the camera (in cm) using a pinhole camera model |
| **Position Tracking** | Reports whether the detected sign is on the LEFT, CENTER, or RIGHT of the frame |
| **DroidCam Support** | Designed to work with a smartphone camera via [DroidCam](https://www.dev47apps.com/) |
| **Live FPS Counter** | On-screen frames-per-second display for performance monitoring |

---

## 📁 Project Structure

```
ARCHITH/
├── traffffic.py              # Main inference & detection script
├── traffic_sign_model.tflite  # Pre-trained TFLite model (~4.4 MB)
└── README.md                  # You are here
```

---

## 🛠️ Prerequisites

- **Python** 3.8+
- A webcam **or** a smartphone with [DroidCam](https://www.dev47apps.com/) installed

### Python Dependencies

```
opencv-python
numpy
tensorflow
```

Install them all at once:

```bash
pip install opencv-python numpy tensorflow
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd ARCHITH
```

### 2. Install dependencies

```bash
pip install opencv-python numpy tensorflow
```

### 3. Connect your camera

- **Webcam** — just plug it in; the script auto-detects camera indices 0–2.
- **DroidCam** — open the DroidCam app on your phone **and** the DroidCam PC client on your computer, then start streaming.

### 4. Run the detector

```bash
python traffffic.py
```

A window titled **"Traffic Sign Detection"** will open. Point the camera at a traffic sign and watch it get classified in real-time.

Press **`q`** to quit.

---

## 🧠 How It Works

```
Camera Frame
     │
     ▼
┌─────────────────────┐
│  HSV Color Filtering │  ← Red / Blue / Yellow masks
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Contour Detection   │  ← Area & aspect-ratio filtering
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  ROI Extraction      │  ← Padded bounding box crop
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  TFLite Inference    │  ← MobileNetV2, 224×224 input
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Overlay Results     │  ← Label, confidence, distance, position
└─────────────────────┘
```

1. **Color Segmentation** — The frame is converted to HSV and filtered for red, blue, and yellow regions (the dominant colors of traffic signs).
2. **Contour Analysis** — Morphological operations clean the mask, then contours are filtered by area (800–150 000 px²) and aspect ratio (0.4–2.5).
3. **Classification** — Each candidate region is resized to 224×224, normalized, and fed into the TFLite interpreter. Predictions above a **50% confidence threshold** are displayed.
4. **Distance & Position** — Distance is estimated using a simple pinhole model (`distance = (known_width × focal_length) / pixel_width`). Position is determined by which horizontal third of the frame the sign center falls in.

---

## ⚙️ Configuration

Key constants you can tweak in `traffffic.py`:

| Constant | Default | Description |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.50` | Minimum confidence to display a prediction |
| `KNOWN_SIGN_WIDTH_CM` | `60` | Assumed real-world sign width (cm) for distance calc |
| `FOCAL_LENGTH` | `500` | Camera focal length estimate (pixels) |
| `CAP_PROP_FRAME_WIDTH` | `640` | Capture resolution width |
| `CAP_PROP_FRAME_HEIGHT` | `480` | Capture resolution height |

---

## 🏷️ Supported Sign Classes (43)

<details>
<summary>Click to expand full class list</summary>

| ID | Sign |
|---|---|
| 0 | Speed limit (20 km/h) |
| 1 | Speed limit (30 km/h) |
| 2 | Speed limit (50 km/h) |
| 3 | Speed limit (60 km/h) |
| 4 | Speed limit (70 km/h) |
| 5 | Speed limit (80 km/h) |
| 6 | End of speed limit (80 km/h) |
| 7 | Speed limit (100 km/h) |
| 8 | Speed limit (120 km/h) |
| 9 | No passing |
| 10 | No passing for heavy vehicles |
| 11 | Right-of-way at intersection |
| 12 | Priority road |
| 13 | Yield |
| 14 | Stop |
| 15 | No vehicles |
| 16 | Heavy vehicles prohibited |
| 17 | No entry |
| 18 | General caution |
| 19 | Dangerous curve left |
| 20 | Dangerous curve right |
| 21 | Double curve |
| 22 | Bumpy road |
| 23 | Slippery road |
| 24 | Road narrows right |
| 25 | Road work |
| 26 | Traffic signals |
| 27 | Pedestrians |
| 28 | Children crossing |
| 29 | Bicycles crossing |
| 30 | Beware of ice/snow |
| 31 | Wild animals crossing |
| 32 | End of all limits |
| 33 | Turn right |
| 34 | Turn left |
| 35 | Ahead only |
| 36 | Go right |
| 37 | Go straight or left |
| 38 | Keep right |
| 39 | Keep left |
| 40 | Roundabout mandatory |
| 41 | End of no passing |
| 42 | End of no passing by heavy vehicles |

</details>

---

## 🤝 Author

**Archith** — [Portfolio](https://archith-portfolio-omega.vercel.app/)

---

## 📜 License

This project is for educational purposes (4th Semester PBL-2). Feel free to use and modify with attribution.
