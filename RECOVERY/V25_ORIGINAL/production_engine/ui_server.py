import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from production_runner import run_production_localization

app = Flask(__name__)
CORS(app)  # Enable CORS for UI integration

@app.route("/predict", methods=["POST"])
def predict():
    if "reference" not in request.files or "search" not in request.files:
        return jsonify({"error": "Missing reference or search image files"}), 400
        
    ref_file = request.files["reference"]
    search_file = request.files["search"]
    
    # Read files
    ref_bytes = ref_file.read()
    search_bytes = search_file.read()
    
    ref_np = np.frombuffer(ref_bytes, np.uint8)
    search_np = np.frombuffer(search_bytes, np.uint8)
    
    ref_img = cv2.imdecode(ref_np, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imdecode(search_np, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None or search_img is None:
        return jsonify({"error": "Could not decode uploaded images"}), 400
        
    # Run production runner localization
    try:
        res = run_production_localization(ref_img, search_img, verbose=True)
    except Exception as e:
        return jsonify({"error": f"Error running production localization pipeline: {str(e)}"}), 500
        
    # Generate visualization overlay if target is found
    overlay_b64 = ""
    if res["found"] == 1:
        # Load color copy of search image
        search_color = cv2.imdecode(search_np, cv2.IMREAD_COLOR)
        
        # Calculate bounding box corners based on (x, y), scale, and rotation
        h_ref, w_ref = ref_img.shape[:2]
        s = res["scale"]
        theta = res["theta"]
        
        tw = int(round(w_ref / s))
        th = int(round(h_ref / s))
        
        cx, cy = res["x"], res["y"]
        
        # Define 4 template box corners relative to center
        corners = np.array([
            [-tw / 2.0, -th / 2.0],
            [tw / 2.0, -th / 2.0],
            [tw / 2.0, th / 2.0],
            [-tw / 2.0, th / 2.0]
        ])
        
        # Apply CCW rotation matrix
        angle_rad = np.radians(theta)
        c, n_s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -n_s], [n_s, c]])
        
        rotated_corners = corners @ R.T
        
        # Translate to predicted coordinates
        final_corners = rotated_corners + np.array([cx, cy])
        final_corners = final_corners.astype(np.int32)
        
        # Draw bounding box
        cv2.polylines(search_color, [final_corners], isClosed=True, color=(0, 255, 0), thickness=3)
        
        # Draw center point
        cv2.circle(search_color, (int(round(cx)), int(round(cy))), radius=6, color=(0, 0, 255), thickness=-1)
        
        # Encode overlay image to base64
        _, buffer = cv2.imencode(".png", search_color)
        overlay_b64 = base64.b64encode(buffer).decode("utf-8")
        
    res["overlay_b64"] = overlay_b64
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
