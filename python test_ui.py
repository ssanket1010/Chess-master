import os
import cv2
import argparse
from ultralytics import YOLO
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="Chess Vision Test Static Diagnostic Runner")
parser.add_argument("--model", default=os.path.join(BASE_DIR, "best.pt"), help="Path to best.pt")
parser.add_argument("--image", default=os.path.join(BASE_DIR, "chess.png"), help="Path to static verification asset image")
args = parser.parse_args()

MODEL_PATH = args.model
IMAGE_PATH = args.image

app = Flask(__name__)
CORS(app)

# Ensure checking logic runs gracefully
model = YOLO(MODEL_PATH)

def box_to_square(x1, y1, x2, y2):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    col = min(int(cx / 800 * 8), 7)
    row = min(int(cy / 800 * 8), 7)
    return col, row

def detect_on_image():
    if not os.path.exists(IMAGE_PATH):
        return [{"error": f"Asset target not found at {IMAGE_PATH}. Please provide a valid evaluation file."}]
        
    image = cv2.imread(IMAGE_PATH)
    image = cv2.resize(image, (800, 800))
    results = model(image)[0]
    pieces = []

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = box
            label = model.names[int(cls)]
            col, row = box_to_square(x1, y1, x2, y2)

            pieces.append({
                "label": label,
                "col": col,
                "row": row,
                "confidence": float(conf)
            })
    return pieces

@app.route("/pieces")
def get_pieces():
    return jsonify(detect_on_image())

@app.route("/")
def index():
    return render_template_string("""
    <html>
        <head><title>Chess Vision — Static Diagnostic Endpoint Mode</title></head>
        <body style="font-family:sans-serif; padding:40px; background:#0f1115; color:#e2e8f0;">
            <h2>Static Diagnostic Endpoint Running</h2>
            <p>Target Evaluation Weight Asset Path: <code>{{ m_path }}</code></p>
            <p>Target Static Evaluation Image File: <code>{{ i_path }}</code></p>
            <p>Hit JSON tracking payload outputs directly here: <a href="/pieces" style="color:#f0c040;">/pieces</a></p>
        </body>
    </html>
    """, m_path=MODEL_PATH, i_path=IMAGE_PATH)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)