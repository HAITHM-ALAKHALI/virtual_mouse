import cv2
import mediapipe as mp
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.maxHands = maxHands
        self.lmList = []
        self.results = None

        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        self.results = self.detector.detect(mpImage)
        if self.results.hand_landmarks:
            for handLms in self.results.hand_landmarks:
                if draw:
                    h, w, c = img.shape
                    points = []
                    for lm in handLms:
                        cx = int(lm.x * w)
                        cy = int(lm.y * h)
                        points.append((cx, cy))
                    for point in points:
                        cv2.circle(img, point, 5, (255, 0, 255), cv2.FILLED)
                    connections = [
                        (0, 1),
                        (1, 2),
                        (2, 3),
                        (3, 4),
                        (0, 5),
                        (5, 6),
                        (6, 7),
                        (7, 8),
                        (5, 9),
                        (9, 10),
                        (10, 11),
                        (11, 12),
                        (9, 13),
                        (13, 14),
                        (14, 15),
                        (15, 16),
                        (13, 17),
                        (17, 18),
                        (18, 19),
                        (19, 20),
                        (0, 17),
                    ]
                    for start, end in connections:
                        cv2.line(img, points[start], points[end], (0, 255, 0), 2)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []
        if self.results and self.results.hand_landmarks:
            myHand = self.results.hand_landmarks[handNo]
            h, w, c = img.shape
            xList = []
            yList = []
            for id, lm in enumerate(myHand):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                # if draw:
                #     cv2.circle(
                #         img,
                #         (cx, cy),
                #         5,
                #         (255,0,255),
                #         cv2.FILLED
                #     )
            xmin = min(xList)
            xmax = max(xList)
            ymin = min(yList)
            ymax = max(yList)
            bbox = [xmin, ymin, xmax, ymax]
        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if len(self.lmList) == 0:
            return []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Four fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1 = self.lmList[p1][1]
        y1 = self.lmList[p1][2]
        x2 = self.lmList[p2][1]
        y2 = self.lmList[p2][2]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
