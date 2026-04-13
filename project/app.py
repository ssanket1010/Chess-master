import cv2
import numpy as np
import os
from pathlib import Path
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "best.pt"))
IP_URL = os.getenv("IP_URL", "0")
BOARD_SIZE = 800  # The pixel size of our warped board

model = YOLO(MODEL_PATH)

# Mapping YOLO labels to Chessboard.js codes (w=White, b=Black)
# Update these keys to match your YOLO 'model.names' exactly!
PIECE_MAP = {
    "white-pawn": "wP", "white-rook": "wR", "white-knight": "wN", 
    "white-bishop": "wB", "white-king": "wK", "white-queen": "wQ",
    "black-pawn": "bP", "black-rook": "bR", "black-knight": "bN", 
    "black-bishop": "bB", "black-king": "bK", "black-queen": "bQ"
}

# --- Core Logic Functions ---

def detect_board(image):
    """Finds 7x7 internal corners and returns the 4 outer corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pattern_size = (7, 7)

    # More robust corner detector (OpenCV 4.5+)
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, None)
    if not ret:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not ret:
        return None

    corners = corners.reshape(-1, 2)
    # 7x7 grid: 0 is top-left, 6 is top-right, 42 is bottom-left, 48 is bottom-right
    tl, tr, bl, br = corners[0], corners[6], corners[-7], corners[-1]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp_board(image, pts):
    """Perspective transform to a flat 800x800 square."""
    dst = np.array([
        [0, 0], [BOARD_SIZE-1, 0], 
        [BOARD_SIZE-1, BOARD_SIZE-1], [0, BOARD_SIZE-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (BOARD_SIZE, BOARD_SIZE))

def get_chess_coords(x, y):
    """
    Converts pixel coordinates on the 800x800 warped image to algebraic notation.
    Math: $col = \lfloor \frac{x}{100} \rfloor$, $row = 8 - \lfloor \frac{y}{100} \rfloor$
    """
    columns = "abcdefgh"
    cell_size = BOARD_SIZE / 8
    col_idx = int(x // cell_size)
    row_idx = 8 - int(y // cell_size)
    # Boundary checks
    col_idx = max(0, min(7, col_idx))
    row_idx = max(1, min(8, row_idx))
    return f"{columns[col_idx]}{row_idx}"

# --- Flask Streaming Logic ---

def gen_frames():
    stream_source = int(IP_URL) if IP_URL.isdigit() else IP_URL
    cap = cv2.VideoCapture(stream_source)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. Perspective Correction
        board_pts = detect_board(frame)
        current_state = app.config.get('LATEST_STATE', {}).copy()

        if board_pts is not None:
            current_state = {}
            # Draw board polygon on the original frame for easier debugging
            cv2.polylines(frame, [board_pts.astype(np.int32)], True, (0, 255, 255), 2)
            warped = warp_board(frame, board_pts)
            
            # 2. Detect pieces on the warped board
            results = model(warped, verbose=False)[0]
            
            if results.boxes:
                for box, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = box
                    # Calculate center of the base of the piece
                    center_x = (x1 + x2) / 2
                    center_y = y2  # Use bottom of box for more accurate square detection
                    
                    label = model.names[int(cls)]
                    square = get_chess_coords(center_x, center_y)
                    
                    if label in PIECE_MAP:
                        current_state[square] = PIECE_MAP[label]

                    # Draw on the warped frame for the UI feed
                    cv2.rectangle(warped, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(warped, label, (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            display_frame = warped
        else:
            # If board not detected, show original frame with a warning
            display_frame = frame
            cv2.putText(display_frame, "Board Not Detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update global state for the JSON API
        app.config['LATEST_STATE'] = current_state

        # Encode for web
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/board_state')
def board_state():
    # Return the dictionary of { "e4": "wP", ... }
    return jsonify(app.config.get('LATEST_STATE', {}))

if __name__ == '__main__':
    # Using threaded=True to handle multiple requests (video + state polling)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
