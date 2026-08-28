"""
HandTrackingModule — MediaPipe Hand Landmark Detection Wrapper

Provides a high-level interface for detecting and tracking hand landmarks
using the MediaPipe HandLandmarker task API. Supports drawing, finger-state
detection, and inter-landmark distance measurement.
"""

import cv2
import mediapipe as mp
import math
import os
import logging
from typing import List, Tuple, Optional

__all__ = ["handDetector"]

logger = logging.getLogger(__name__)


class handDetector:
    """Detects and tracks hand landmarks in video frames.

    Uses the MediaPipe HandLandmarker task model to locate 21 hand landmarks
    per detected hand.  Offers convenience helpers for finger-up detection,
    bounding-box extraction, and inter-landmark distance.

    Args:
        mode:         Not used by the task API; kept for backward compatibility.
        maxHands:     Maximum number of hands to detect (default 2).
        detectionCon: Minimum confidence for initial hand detection (0–1).
        trackCon:     Minimum confidence for hand tracking between frames (0–1).
    """

    # Landmark indices for each fingertip
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
            raise FileNotFoundError(
                f"Hand landmarker model not found at: {model_path}"
            )

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
            maxHands, detectionCon, trackCon,
        )

    # ------------------------------------------------------------------
    # Detection & Drawing
    # ------------------------------------------------------------------

    def findHands(self, img, draw: bool = True):
        """Detect hands and optionally draw landmarks + skeleton on *img*.

        Args:
            img:  BGR image (numpy array from OpenCV).
            draw: If True, draw landmarks and connections on *img* in-place.

        Returns:
            The (possibly annotated) image.
        """
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
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (5, 9), (9, 10), (10, 11), (11, 12),
                        (9, 13), (13, 14), (14, 15), (15, 16),
                        (13, 17), (17, 18), (18, 19), (19, 20),
                        (0, 17),
                    ]
                    for start, end in connections:
                        cv2.line(img, points[start], points[end], (0, 255, 0), 2)
        return img

    # ------------------------------------------------------------------
    # Position & Bounding Box
    # ------------------------------------------------------------------

    def findPosition(
        self, img, handNo: int = 0, draw: bool = True
    ) -> Tuple[List[List[int]], List[int]]:
        """Extract landmark positions and bounding box for one hand.

        Args:
            img:    BGR image used to scale normalised coordinates.
            handNo: Index of the hand to query (0-based).
            draw:   (reserved for future use — currently unused).

        Returns:
            A tuple ``(lmList, bbox)`` where:
            - *lmList* is ``[[id, cx, cy], ...]`` for 21 landmarks, or ``[]``.
            - *bbox* is ``[xmin, ymin, xmax, ymax]``, or ``[]``.
        """
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
            if self.results and self.results.hand_landmarks and handNo >= len(self.results.hand_landmarks):
                logger.warning(
                    "Requested handNo=%d but only %d hand(s) detected",
                    handNo, len(self.results.hand_landmarks),
                )

        return self.lmList, bbox

    # ------------------------------------------------------------------
    # Finger State
    # ------------------------------------------------------------------

    def fingersUp(self) -> List[int]:
        """Return a list of 5 ints (0 or 1) indicating which fingers are up.

        Order: [Thumb, Index, Middle, Ring, Pinky].
        Returns an empty list if no landmarks are available.
        """
        fingers: List[int] = []
        if len(self.lmList) == 0:
            return []

        # Thumb — compare x-coordinates (works for right hand facing camera)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Index → Pinky — compare y-coordinates (lower y = higher on screen)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def findDistance(
        self, p1: int, p2: int, img, draw: bool = True
    ) -> Tuple[float, any, List[int]]:
        """Compute the Euclidean distance between two landmarks.

        Args:
            p1, p2: Landmark indices (0–20).
            img:    Image to draw on.
            draw:   If True, draw circles and connecting line.

        Returns:
            ``(length, img, [x1, y1, x2, y2, cx, cy])``
        """
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
