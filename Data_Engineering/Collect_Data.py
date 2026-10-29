"""
================================================================================
Collect_Data.py — Empirical Gesture Landmark Data Collection Pipeline
Virtual Mouse Control System
Paper Reference: Section III-B ("Bifurcated Feature Normalization Pipeline")
                 Section IV-A ("Dataset Formulation and Baseline Benchmark")
================================================================================

Description:
    Captures real-time monocular hand landmark streams, computes the 42-dimensional
    anatomically normalized feature representation (Eq. 5–8 in paper), and appends
    labeled training instances to `virtual_mouse_data.csv`.

Mathematical Pipeline (Section III-B):
    1. Translation Normalization (Eq. 5):   q_i = p_i - p_0  (wrist root origin)
    2. Anatomical Scale Factor (Eq. 6):     s = ||p_9 - p_0||_2  (wrist to middle MCP)
    3. Scale Normalization (Eq. 7):         \hat{q}_i = q_i / max(s, \epsilon)
    4. Handedness Canonicalization (Eq. 8): \hat{x}_i <- -\hat{x}_i  (if Left hand)

Operational Gesture Classes (Section IV-A):
    'm' -> Move         | 'l' -> Left_Click   | 'r' -> Right_Click
    's' -> Scroll       | 'd' -> Drag         | 'n' -> Neutral
    'q' -> Terminate acquisition loop
================================================================================
"""

import sys
import os
import math

# -----------------------------------------------------------------------------
# Module Search Path Configuration
# -----------------------------------------------------------------------------
# Resolves dynamic root path to import HandTrackingModule from Utils directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import Utils.HandTrackingModule as htm
import cv2
import csv

# -----------------------------------------------------------------------------
# Dataset Target Resolution & CSV Header Initialization
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "virtual_mouse_data.csv")

# Build 42D normalized landmark coordinate header: [lm_0_x, lm_0_y, ..., lm_20_y]
header = []
for i in range(21):
    header.extend([f"lm_{i}_x", f"lm_{i}_y"])
header.append("gesture_label")

# Initialize dataset file with header row if not already present on disk
file_exists = os.path.isfile(csv_path)
if not file_exists:
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# -----------------------------------------------------------------------------
# Video Capture & Perception Module Initialization
# -----------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1)

# -----------------------------------------------------------------------------
# Continuous Acquisition and Normalization Loop
# -----------------------------------------------------------------------------
while True:
    success, img = cap.read()
    if not success:
        break
        
    # Horizontal flip to provide intuitive mirror-like user feedback
    img = cv2.flip(img, 1)

    # Detect hand landmarks using MediaPipe Tasks pipeline
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        row_data = []
        
        # ---------------------------------------------------------------------
        # 1. Translation Normalization (Wrist Landmark p0 as Origin - Eq. 5)
        # ---------------------------------------------------------------------
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]

        # ---------------------------------------------------------------------
        # 2. Anatomical Scale Factor Calculation (Wrist to Middle MCP - Eq. 6)
        # ---------------------------------------------------------------------
        middle_mcp_x = lmList[9][1] - wrist_x
        middle_mcp_y = lmList[9][2] - wrist_y
        scale = math.sqrt(middle_mcp_x**2 + middle_mcp_y**2)

        # Zero-scale singularity safeguard (Equation 7 guard)
        if scale != 0:
            # -----------------------------------------------------------------
            # 3. Handedness Canonicalization Mapping (Section III-B & Eq. 8)
            # -----------------------------------------------------------------
            handedness = detector.getHandedness()

            # Account for horizontal frame flip in camera coordinate space
            if handedness == "Right":
                actual_hand = "Left"
            elif handedness == "Left":
                actual_hand = "Right"
            else:
                actual_hand = None

            # -----------------------------------------------------------------
            # 4. Feature Vector Normalization (Equations 7 & 8)
            # -----------------------------------------------------------------
            for lm in lmList:
                norm_x = (lm[1] - wrist_x) / scale
                norm_y = (lm[2] - wrist_y) / scale

                # Horizontal reflection: Canonicalize Left hand into Right hand space
                if actual_hand == "Left":
                    norm_x = -norm_x

                row_data.extend([norm_x, norm_y])

    cv2.imshow("Data Collector", img)
    key = cv2.waitKey(1) & 0xFF

    # -------------------------------------------------------------------------
    # Label Binding & Sample Persistence (Paper Section IV-A)
    # -------------------------------------------------------------------------
    if len(lmList) != 0:
        label = ""
        if key == ord("m"):
            label = "Move"
        elif key == ord("l"):
            label = "Left_Click"
        elif key == ord("r"):
            label = "Right_Click"
        elif key == ord("s"):
            label = "Scroll"
        elif key == ord("d"):
            label = "Drag"
        elif key == ord("n"):
            label = "Neutral"

        # Append complete 42D sample + ground truth label to dataset
        if label != "":
            row_data.append(label)
            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow(row_data)

    # Exit acquisition loop
    if key == ord("q"):
        break

# Release hardware capture resources
cap.release()
cv2.destroyAllWindows()