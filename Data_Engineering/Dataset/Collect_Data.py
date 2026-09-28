"""
Collect_Data — Gesture Landmark Data Collection Pipeline

Captures 42-dimensional normalised landmark feature vectors from a live
webcam feed and appends labelled rows to ``virtual_mouse_data.csv``.

Normalisation steps (applied identically at inference time in Main.py):
  1. Translation — wrist landmark (p0) is set as the coordinate origin.
  2. Scale — Euclidean distance from wrist to middle-MCP joint (p9) is
     used as the divisor for approximate scale invariance.
  3. Handedness — x-coordinates are horizontally reflected (x <- -x)
     when the detected physical hand is the left hand.

Key bindings:
  m -> Move | l -> Left_Click | r -> Right_Click
  s -> Scroll | d -> Drag | n -> Neutral | q -> Quit
"""

import sys
import os
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import Utils.HandTrackingModule as htm
import cv2
import csv

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "virtual_mouse_data.csv")

header = []

for i in range(21):
    header.extend([f"lm_{i}_x", f"lm_{i}_y"])
header.append("gesture_label")

file_exists = os.path.isfile(csv_path)

if not file_exists:
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        row_data = []
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]

        middle_mcp_x = lmList[9][1] - wrist_x
        middle_mcp_y = lmList[9][2] - wrist_y

        scale = math.sqrt(middle_mcp_x**2 + middle_mcp_y**2)

        if scale != 0:
            handedness = detector.getHandedness()

            if handedness == "Right":
                actual_hand = "Left"
            elif handedness == "Left":
                actual_hand = "Right"
            else:
                actual_hand = None

            for lm in lmList:
                norm_x = (lm[1] - wrist_x) / scale
                norm_y = (lm[2] - wrist_y) / scale

                if actual_hand == "Left":
                    norm_x = -norm_x

                row_data.extend([norm_x, norm_y])

    cv2.imshow("Data Collector", img)
    key = cv2.waitKey(1) & 0xFF

    if len(lmList) != 0:
        label = ""
        if key == ord("m"): # ASCII
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

        if label != "":
            row_data.append(label)
            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow(row_data)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
