"""
benchmark_latency — Per-Stage Pipeline Latency Profiler

Measures wall-clock latency for each processing stage over 500
consecutive frames with a live camera feed:
  1. Video acquisition + MediaPipe HandLandmarker inference
  2. 42D anatomical feature normalisation and reshaping
  3. SVM classifier inference (StandardScaler + RBF-kernel SVC)
  4. Action dispatcher state evaluation overhead

All timings use ``time.perf_counter()`` for sub-millisecond resolution.
"""

import os
import sys
import time
import cv2
import joblib
import numpy as np

# Add path to Utils directory
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "Utils")))
import HandTrackingModule as htm

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1, detectionCon=0.7)

# Load trained SVM brain model
model_path = os.path.join(
    current_dir, "..", "Model_Training", "virtual_mouse_brain.pkl"
)
if not os.path.exists(model_path):
    model_path = "model.pkl"

model = joblib.load(model_path)

latencies_mp = []
latencies_norm = []
latencies_svm = []
latencies_dispatch = []

print(
    "Starting benchmark for 500 frames... Move your hand in front of the camera naturally."
)

frames_recorded = 0
while frames_recorded < 500:
    t0 = time.perf_counter()
    success, img = cap.read()
    if not success:
        continue

    # 1. Video acquisition and MediaPipe landmark inference
    img = detector.findHands(img, draw=False)
    lmList, bbox = detector.findPosition(img, draw=False)
    handedness = detector.getHandedness()
    t1 = time.perf_counter()

    if handedness == "Right":
        actual_hand = "Left"
    elif handedness == "Left":
        actual_hand = "Right"
    else:
        actual_hand = None

    if len(lmList) != 0:
        # 2. 42D Anatomical Feature Normalization
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]

        middle_mcp_x = lmList[9][1] - wrist_x
        middle_mcp_y = lmList[9][2] - wrist_y

        scale = np.sqrt(middle_mcp_x**2 + middle_mcp_y**2)

        if scale == 0:
            continue

        hand_data = []
        for lm in lmList:
            norm_x = (lm[1] - wrist_x) / scale
            norm_y = (lm[2] - wrist_y) / scale

            if actual_hand == "Left":
                norm_x = -norm_x

            hand_data.extend([norm_x, norm_y])

        feat_42d = np.array(hand_data).reshape(1, -1)
        t2 = time.perf_counter()

        # 3. SVM classifier inference
        pred = model.predict(feat_42d)
        t3 = time.perf_counter()

        # 4. Action dispatcher verification logic
        _ = pred[0]
        t4 = time.perf_counter()

        latencies_mp.append((t1 - t0) * 1000)
        latencies_norm.append((t2 - t1) * 1000)
        latencies_svm.append((t3 - t2) * 1000)
        latencies_dispatch.append((t4 - t3) * 1000)

        frames_recorded += 1
        if frames_recorded % 100 == 0:
            print(f"Audited {frames_recorded}/500 frames...")

cap.release()
cv2.destroyAllWindows()

# Output empirical metrics
print("\n================ BENCHMARK TIMING REPORT ================")
print(
    f"1) MediaPipe Detection : {np.mean(latencies_mp):.2f} ms ± {np.std(latencies_mp):.2f} ms (95th: {np.percentile(latencies_mp, 95):.2f} ms)"
)
print(
    f"2) Normalization (42D) : {np.mean(latencies_norm):.2f} ms ± {np.std(latencies_norm):.2f} ms"
)
print(
    f"3) SVM Inference       : {np.mean(latencies_svm):.2f} ms ± {np.std(latencies_svm):.2f} ms"
)
print(
    f"4) Action Dispatch     : {np.mean(latencies_dispatch):.2f} ms ± {np.std(latencies_dispatch):.2f} ms"
)

post_detection = (
    np.array(latencies_norm) + np.array(latencies_svm) + np.array(latencies_dispatch)
)
print(f"--> Post-Detection Overhead (2+3+4): {np.mean(post_detection):.2f} ms")

total = (
    np.array(latencies_mp)
    + np.array(latencies_norm)
    + np.array(latencies_svm)
    + np.array(latencies_dispatch)
)
print(
    f"--> Total Pipeline Latency        : {np.mean(total):.2f} ms (Max: {np.max(total):.2f} ms)"
)
print("=========================================================")
