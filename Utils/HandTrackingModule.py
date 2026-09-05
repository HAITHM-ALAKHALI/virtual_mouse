import cv2
import mediapipe as mp
import math
import os
import logging
from typing import List, Tuple, Optional
import time

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

        self.mp_vision = mp_vision

        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )

        self.detector = mp_vision.HandLandmarker.create_from_options(options)

        self.tipIds = self.TIP_IDS

        self.target_wrist: Optional[Tuple[float, float]] = None

        self.target_hand = None

        self.selected_hand = None

        self.lost_frames = 0

        self.max_lost_frames = 20

        self.max_wrist_distance = 0.25

        self.start_time = time.time()
        self.timestamp_ms = 0

        logger.debug(
            "handDetector initialized (maxHands=%d, detectionCon=%.2f, trackCon=%.2f)",
            maxHands,
            detectionCon,
            trackCon,
        )

    def _wrist_distance(self, hand) -> float:

        if self.target_wrist is None:
            return float("inf")

        wrist = hand[0]

        dx = wrist.x - self.target_wrist[0]
        dy = wrist.y - self.target_wrist[1]

        return math.hypot(dx, dy)

    def _selectTargetHand(self):

        if self.results is None:
            return None

        hands = self.results.hand_landmarks

        if not hands:

            if self.target_wrist is not None:
                self.lost_frames += 1

                if self.lost_frames > self.max_lost_frames:
                    self.target_wrist = None
                    self.target_hand = None

            return None

        if self.target_wrist is None:

            selected = hands[0]

            wrist = selected[0]

            self.target_wrist = (wrist.x, wrist.y)

            self.target_hand = selected

            self.lost_frames = 0

            return selected

        closest_hand = None
        closest_distance = float("inf")

        for hand in hands:

            distance = self._wrist_distance(hand)

            if distance < closest_distance:
                closest_distance = distance
                closest_hand = hand

        if closest_distance <= self.max_wrist_distance:

            selected = closest_hand

            wrist = selected[0]

            self.target_wrist = (wrist.x, wrist.y)

            self.target_hand = selected

            self.lost_frames = 0

            return selected

        self.lost_frames += 1

        if self.lost_frames > self.max_lost_frames:

            self.target_wrist = None
            self.target_hand = None
            self.lost_frames = 0

        return None

    def findHands(self, img, draw: bool = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        self.timestamp_ms = int((time.time() - self.start_time) * 1000)

        self.results = self.detector.detect_for_video(mpImage, self.timestamp_ms)

        self.selected_hand = self._selectTargetHand()

        if self.selected_hand is None:
            return img

        if draw:

            h, w, c = img.shape

            points = []

            for lm in self.selected_hand:

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

        myHand = self.selected_hand

        if myHand is None:
            return self.lmList, bbox

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

        return self.lmList, bbox

    def getHandedness(self, handNo: int = 0) -> Optional[str]: # right hand or left hand 

        if self.selected_hand is None:
            return None

        if self.results and self.results.handedness:

            for i, hand in enumerate(self.results.hand_landmarks):

                if hand is self.selected_hand:

                    return self.results.handedness[i][0].category_name

        return None

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

        return (length, img, [x1, y1, x2, y2, cx, cy])

    def resetTarget(self):

        self.target_wrist = None
        self.target_hand = None
        self.selected_hand = None
        self.lost_frames = 0

    def close(self):

        if self.detector is not None:
            self.detector.close()
