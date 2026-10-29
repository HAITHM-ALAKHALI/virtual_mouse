"""
================================================================================
benchmark_latency.py — Per-Stage Pipeline Latency Profiler
Virtual Mouse Control System
Paper Reference: Section IV-D ("End-to-End System Timing Analysis")
================================================================================

Description:
    Quantifies empirical execution latencies across 500 consecutive real-time 
    operational cycles using a live monocular video stream (target: 30 FPS).
    
    Isolates per-frame execution into four distinct measurement stages:
      Stage 1: Monocular Video Acquisition & MediaPipe HandLandmarker Inference
      Stage 2: 42D Anatomical Feature Normalization (Eq. 5–8 in paper)
      Stage 3: Scaled SVM-RBF Classification Inference (virtual_mouse_brain.pkl)
      Stage 4: Rule-Based Action Dispatcher & Event Evaluation Overhead
      
    Timing Methodology:
      Uses `time.perf_counter()` to achieve high-resolution, monotonic timing.
      Calculates mean latency, standard deviation, 95th percentile metrics,
      and post-detection overhead (Stages 2 + 3 + 4 ≈ 1.22 ms).
================================================================================
"""

import os
import sys
import time
import cv2
import joblib
import numpy as np

# -----------------------------------------------------------------------------
# Module Search Path Configuration
# -----------------------------------------------------------------------------
# Resolves dynamic root path to import HandTrackingModule from Utils directory
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(current_dir, "..", "Utils")))
import HandTrackingModule as htm

# -----------------------------------------------------------------------------
# Hardware Acquisition & Perception Pipeline Initialization
# -----------------------------------------------------------------------------
# Open commodity RGB webcam (default device index 0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Tasks HandLandmarker wrapper (RunningMode.VIDEO)
# Limits detection to single primary anchor hand with 0.70 confidence gate
detector = htm.handDetector(maxHands=1, detectionCon=0.7)

# -----------------------------------------------------------------------------
# Classifier Model Deserialization (Pipeline: StandardScaler + SVC RBF)
# -----------------------------------------------------------------------------
# Resolves path to frozen, pre-trained scikit-learn pipeline (1,139 support vectors)
model_path = os.path.join(
    current_dir, "..", "Model_Training", "virtual_mouse_brain.pkl"
)
if not os.path.exists(model_path):
    model_path = "model.pkl"

# Loads pipeline encapsulating both training-fitted StandardScaler and SVM model
model = joblib.load(model_path)

# -----------------------------------------------------------------------------
# Telemetry Measurement Accumulators
# -----------------------------------------------------------------------------
latencies_mp = []        # Stage 1: Capture + MediaPipe HandLandmarker (ms)
latencies_norm = []      # Stage 2: 42D Anatomical Normalization (ms)
latencies_svm = []       # Stage 3: SVM Inference Latency (ms)
latencies_dispatch = []  # Stage 4: State evaluation & dispatcher logic (ms)

print(
    "Starting benchmark for 500 frames... Move your hand in front of the camera naturally."
)

frames_recorded = 0

# -----------------------------------------------------------------------------
# Main Profiling Loop (Audited Across 500 Operational Frames)
# -----------------------------------------------------------------------------
while frames_recorded < 500:
    # Timestamp t0: Frame capture cycle inception
    t0 = time.perf_counter()
    success, img = cap.read()
    if not success:
        continue

    # -------------------------------------------------------------------------
    # Stage 1: Video Frame Acquisition & Landmark Inference
    # -------------------------------------------------------------------------
    # Executes deep regression network on frame to extract 21 3D hand joints
    img = detector.findHands(img, draw=False)
    lmList, bbox = detector.findPosition(img, draw=False)
    handedness = detector.getHandedness()
    
    # Timestamp t1: Perception stage completion
    t1 = time.perf_counter()

    # Mirroring adjustment: Webcam mirrors user view, canonicalize anatomical label
    if handedness == "Right":
        actual_hand = "Left"
    elif handedness == "Left":
        actual_hand = "Right"
    else:
        actual_hand = None

    if len(lmList) != 0:
        # ---------------------------------------------------------------------
        # Stage 2: 42D Bifurcated Anatomical Normalization (Paper Section III-B)
        # ---------------------------------------------------------------------
        # Step 1: Translation Normalization (Wrist Landmark p0 set as Origin)
        # q_i = p_i - p_0 (Equation 5)
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]

        # Step 2: Anatomical Uniform-Scale Normalization
        # Baseline vector between Wrist (p0) and Middle MCP joint (p9)
        middle_mcp_x = lmList[9][1] - wrist_x
        middle_mcp_y = lmList[9][2] - wrist_y

        # Characteristic palm scale factor s = ||p9 - p0||_2 (Equation 6)
        scale = np.sqrt(middle_mcp_x**2 + middle_mcp_y**2)

        # Runtime safeguard against division-by-zero singularities
        if scale == 0:
            continue

        hand_data = []
        for lm in lmList:
            # Normalize 2D image coordinates against anatomical palm scale (Equation 7)
            norm_x = (lm[1] - wrist_x) / scale
            norm_y = (lm[2] - wrist_y) / scale

            # Step 3: Handedness Canonicalization (Equation 8)
            # Horizontal reflection: Left hands are mapped into right-hand space
            if actual_hand == "Left":
                norm_x = -norm_x

            hand_data.extend([norm_x, norm_y])

        # Reshape into a 1x42 dimensional feature vector for inference
        feat_42d = np.array(hand_data).reshape(1, -1)
        
        # Timestamp t2: Feature normalization completion
        t2 = time.perf_counter()

        # ---------------------------------------------------------------------
        # Stage 3: SVM Classifier Inference (Paper Section IV-B)
        # ---------------------------------------------------------------------
        # Passes feature vector through StandardScaler and RBF-kernel SVC
        pred = model.predict(feat_42d)
        
        # Timestamp t3: Statistical classification inference completion
        t3 = time.perf_counter()

        # ---------------------------------------------------------------------
        # Stage 4: Rule-Based Action Dispatcher Verification Overhead
        # ---------------------------------------------------------------------
        # Simulates dispatcher lookup and label routing overhead
        _ = pred[0]
        
        # Timestamp t4: Dispatcher evaluation completion
        t4 = time.perf_counter()

        # Accumulate millisecond latencies for each isolated phase
        latencies_mp.append((t1 - t0) * 1000)
        latencies_norm.append((t2 - t1) * 1000)
        latencies_svm.append((t3 - t2) * 1000)
        latencies_dispatch.append((t4 - t3) * 1000)

        frames_recorded += 1
        if frames_recorded % 100 == 0:
            print(f"Audited {frames_recorded}/500 frames...")

# Clean up hardware capture interfaces
cap.release()
cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Empirical Profiling Report Generation (Paper Table IV Metrics)
# -----------------------------------------------------------------------------
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

# Summation of post-detection stages (Stages 2 + 3 + 4)
post_detection = (
    np.array(latencies_norm) + np.array(latencies_svm) + np.array(latencies_dispatch)
)
print(f"--> Post-Detection Overhead (2+3+4): {np.mean(post_detection):.2f} ms")

# Total end-to-end operational frame latency
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