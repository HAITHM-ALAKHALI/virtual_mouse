import sys
import os

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

        for lm in lmList:
            norm_x = lm[1] - wrist_x
            norm_y = lm[2] - wrist_y
            row_data.extend([norm_x, norm_y])

    cv2.imshow("Data Collector", img)
    key = cv2.waitKey(1) & 0xFF

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

        if label != "":
            row_data.append(label)
            with open(csv_path, mode="a", newline="") as f:
                csv.writer(f).writerow(row_data)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
