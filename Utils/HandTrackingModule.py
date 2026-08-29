import cv2
import mediapipe as mp
import math
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class handDetector:

    TIP_IDS: List[int] = [4, 8, 12, 16, 20]

    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 2,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ) -> None:
        self.maxHands = maxHands
        self.lmList: List[List[int]] = []
        self.results = None

        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Hand landmarker model not found at: {model_path}")

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)
        self.tipIds = self.TIP_IDS  # backward-compat alias

        logger.debug(
            "handDetector initialized (maxHands=%d, detectionCon=%.2f, trackCon=%.2f)",
            maxHands,
            detectionCon,
            trackCon,
        )

    def findHands(self, img, draw: bool = True):

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

    def findPosition(
        self, img, handNo: int = 0, draw: bool = True
    ) -> Tuple[List[List[int]], List[int]]:

        self.lmList = []
        bbox: List[int] = []

        if (
            self.results
            and self.results.hand_landmarks
            and handNo < len(self.results.hand_landmarks)
        ):
            myHand = self.results.hand_landmarks[handNo]
            h, w, c = img.shape
            xList: List[int] = []
            yList: List[int] = []

            for id, lm in enumerate(myHand):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])

            xmin = min(xList)
            xmax = max(xList)
            ymin = min(yList)
            ymax = max(yList)
            bbox = [xmin, ymin, xmax, ymax]
        else:
            if (
                self.results
                and self.results.hand_landmarks
                and handNo >= len(self.results.hand_landmarks)
            ):
                logger.warning(
                    "Requested handNo=%d but only %d hand(s) detected",
                    handNo,
                    len(self.results.hand_landmarks),
                )

        return self.lmList, bbox

    def fingersUp(self) -> List[int]:

        fingers: List[int] = []
        if len(self.lmList) == 0:
            return []

        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(
        self, p1: int, p2: int, img, draw: bool = True
    ) -> Tuple[float, any, List[int]]:

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
