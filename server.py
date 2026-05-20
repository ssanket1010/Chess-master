import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import threading
import time

# ── CONFIGURATION & ENVIRONMENT SETUP ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Chess Vision Server Backend")
parser.add_argument(
    "--model", 
    default=os.path.join(BASE_DIR, "best.pt"), 
    help="Path to YOLO model weights file"
)
parser.add_argument(
    "--source", 
    default=os.environ.get("CAMERA_URL", "http://10.237.190.235:8080/video"), 
    help="Camera stream URL, IP address, or local device index"
)
parser.add_argument(
    "--port", 
    type=int, 
    default=5000, 
    help="Port to run the Flask server on"
)
args = parser.parse_args()

MODEL_PATH = args.model
IP_URL     = args.source

# ── GLOBALS ──────────────────────────────────────────────────────────────────
model           = YOLO(MODEL_PATH)
lock            = threading.Lock()

latest_raw      = None   # JPEG bytes of raw camera frame
latest_warped   = None   # JPEG bytes of warped+annotated frame
detected_pieces = []
corner_pts      = None   # np.float32 [TL, TR, BR, BL] in original frame pixels
frame_size      = (640, 480)  # updated from actual frames


# ── HELPERS ──────────────────────────────────────────────────────────────────

def warp_board(image, pts):
    dst = np.array([[0,0],[799,0],[799,799],[0,799]], dtype="float32")
    M   = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (800, 800))


def run_detection(frame):
    results = model(frame)[0]
    pieces  = []
    out     = frame.copy()

    if results.boxes is not None:
        boxes   = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs   = results.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls)]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            col = min(int(cx / 800 * 8), 7)
            row = min(int(cy / 800 * 8), 7)

            pieces.append({
                "label":      label,
                "col":        col,
                "row":        row,
                "confidence": round(float(conf), 2),
            })

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(out, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

    # Subtle grid overlay
    step = 800 // 8
    for i in range(9):
        cv2.line(out, (i*step, 0), (i*step, 800), (255, 255, 255), 1)
        cv2.line(out, (0, i*step), (800, i*step), (255, 255, 255), 1)

    return out, pieces


def capture_loop():
    """
    Resilient background frame processing loop with automatic retry tracking 
    to handle transient network drops or delayed device startup.
    """
    global latest_raw, latest_warped, detected_pieces, frame_size

    while True:
        # Resolve integers if numeric camera indices are passed as a string configuration
        try:
            source = int(IP_URL)
        except ValueError:
            source = IP_URL

        print(f"[INFO] Attempting connection to stream source: {source}")
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"[WARNING] Cannot connect to camera resource. Retrying in 5 seconds...")
            cap.release()
            time.sleep(5)
            continue

        print("[INFO] Stream linked successfully.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame dropped. Connection interrupted.")
                break

            h, w = frame.shape[:2]
            frame_size = (w, h)

            # Raw frame mapping with active quad layout overlays
            raw_display = frame.copy()
            with lock:
                pts = corner_pts

            if pts is not None:
                poly = pts.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(raw_display, [poly], True, (0, 220, 255), 2)
                for i, p in enumerate(pts.astype(int)):
                    cv2.circle(raw_display, tuple(p), 7, (0, 220, 255), -1)
                    cv2.putText(raw_display, ["TL","TR","BR","BL"][i],
                                (p[0]+8, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 220, 255), 1)

                # Homography Perspective Correction and Object Mapping
                warped            = warp_board(frame, pts)
                annotated, pieces = run_detection(warped)
                _, w_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                warped_bytes = w_buf.tobytes()
            else:
                warped_bytes = None
                pieces       = []

            _, raw_buf = cv2.imencode(".jpg", raw_display, [cv2.IMWRITE_JPEG_QUALITY, 75])

            with lock:
                latest_raw      = raw_buf.tobytes()
                latest_warped   = warped_bytes
                detected_pieces = pieces

            time.sleep(0.03)

        print("[WARNING] Active connection broken. Cleaning up pipeline for refresh loop...")
        cap.release()
        time.sleep(2)


# Initialize isolated tracking thread asset
app = Flask(__name__)
CORS(app)
threading.Thread(target=capture_loop, daemon=True).start()


# ── STREAMING ROUTINES ────────────────────────────────────────────────────────

def gen(get_fn):
    while True:
        with lock:
            frame = get_fn()
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


# ── API ROUTES ────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard_fallback():
    # Helper index fallback to cleanly host template contents directly if accessed via browser root
    try:
        with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Backend active. Main UI asset file missing at /templates/index.html", 404


@app.route("/raw_feed")
def raw_feed():
    return Response(gen(lambda: latest_raw),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/warped_feed")
def warped_feed():
    return Response(gen(lambda: latest_warped),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/frame_size")
def get_frame_size():
    with lock:
        w, h = frame_size
    return jsonify({"width": w, "height": h})


@app.route("/set_corners", methods=["POST"])
def set_corners():
    global corner_pts
    data = request.get_json()
    pts  = data.get("corners")
    if not pts or len(pts) != 4:
        return jsonify({"error": "Need exactly 4 corners"}), 400
    corner_pts = np.array(pts, dtype="float32")
    return jsonify({"status": "ok"})


@app.route("/clear_corners", methods=["POST"])
def clear_corners():
    global corner_pts
    corner_pts = None
    return jsonify({"status": "cleared"})


@app.route("/pieces")
def pieces():
    with lock:
        data = detected_pieces[:]
    return jsonify(data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=True)