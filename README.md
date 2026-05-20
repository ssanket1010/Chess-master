# Chess Vision — Manual Calibration & Real-Time Detection

An intelligent, computer-vision-driven chess assistant that detects physical chess pieces from a live overhead camera feed, applies perspective correction, maps detected pieces onto an interactive virtual chessboard layout, and outputs the detected pieces in a normalized board grid format.

---

## 🚀 Project Overview

**Chess Vision** uses a fine-tuned **YOLO** object detection model to locate and classify individual chess pieces on a board. Because real-world camera installations are rarely perfectly orthogonal, the project features a manual 4-point calibration interface that transforms the skewed camera angle into a perfect, flat $800 \times 800$ pixel top-down square matrix. This transformed perspective is then gridded into an $8 \times 8$ layout to precisely track column and row positions for every piece.

---

## 🛠️ System Architecture

1. **Backend Server (`server.py`)**: A Flask-based web application orchestrating multi-threaded tasks:
   - Connects to an external IP camera stream via OpenCV.
   - Handles manual quad-corner coordinates sent via HTTP POST.
   - Performs perspective-warping transformations using a homography matrix ($3 \times 3$).
   - Runs object inference through the Ultralytics YOLO framework on the warped viewport.
   - Streams matches as dual multipart streams (`/raw_feed`, `/warped_feed`) and exposes structural piece layouts at `/pieces`.
2. **Frontend UI (`templates/index.html`)**: A sleek, minimal web interface built with pure HTML5, modern CSS, and vanilla JS featuring:
   - A crosshair canvas allowing the user to select Top-Left (TL), Top-Right (TR), Bottom-Right (BR), and Bottom-Left (BL) corners interactively.
   - An interactive virtual chessboard dynamically updating state via recursive polling.
   - Detailed piece confidence indicators and localized chess notation translation.
3. **Inference Test Mode (`python test_ui.py`)**: A standalone module designed to validate YOLO inference on a static frame image asset without setting up camera streams.

---

## 📁 Repository Structure

```text
├── best.pt              # Fine-tuned YOLO weight checkpoint file
├── server.py            # Flask production application server & frame capture loop
├── python test_ui.py    # Static image testing utility server
└── templates/
    └── index.html       # Web UI dashboard for calibration and tracking
```
[Chess Vision — Manual Calibration.pdf](https://github.com/user-attachments/files/28056703/Chess.Vision.Manual.Calibration.pdf)
