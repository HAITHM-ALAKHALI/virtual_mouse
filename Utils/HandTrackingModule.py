"""
================================================================================
HandTrackingModule.py — MediaPipe HandLandmarker & Temporal Anchor Tracker
Virtual Mouse Control System
Paper Reference: Section III-A ("Temporal Target-Hand Anchoring")
================================================================================

Description:
    Encapsulates Google MediaPipe Tasks HandLandmarker (`RunningMode.VIDEO`) 
    with a stateful temporal wrist-anchoring algorithm to enforce target-hand 
    continuity in real-time desktop environments.

Key Mechanisms (Section III-A):
    1. Spatial Wrist Anchoring (Eq. 1): Evaluates candidate hands via Euclidean 
       distance d(h) against previous anchor wrist position w_{t-1}.
    2. Spatial Gating Threshold (D_max = 0.25): Filters secondary background 
       occupants while tolerating fast single-hand ballistic movements.
    3. Temporal Loss Hysteresis (M_lost = 20): Maintains anchor state across 
       transient occlusions up to 20 consecutive frames before reset.
================================================================================
"""

import cv2
import mediapipe as mp
import math
import os
import logging
from typing import List, Tuple, Optional
import time
import sys

logger = logging.getLogger(__name__)


class handDetector:
    """
    Real-time hand tracking and anchor-selection module based on MediaPipe Tasks.
    Maintains a continuous single-hand tracking session using wrist kinematics.
    """
    TIP_IDS: List[int] = [4, 8, 12, 16, 20]

    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 2,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ) -> None:
        """
        Initializes the MediaPipe HandLandmarker pipeline in VIDEO mode.

        Args:
            mode: Legacy boolean parameter for backwards compatibility.
            maxHands: Maximum number of hands to detect concurrently (default: 2).
            detectionCon: Minimum confidence threshold for hand detection (default: 0.5).
            trackCon: Minimum confidence threshold for landmark tracking (default: 0.5).
        """
        self.maxHands = maxHands
        self.lmList: List[List[int]] = []
        self.results = None

        # Resolve asset base path (handles both raw source and PyInstaller runtime environments)
        if getattr(sys, "frozen", False):  # Compiled executable
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base_path, "hand_landmarker.task")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Hand landmarker model not found at: {model_path}")

        # MediaPipe Tasks Python API setup
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        self.mp_vision = mp_vision

        # Configure video-mode landmark inference (RunningMode.VIDEO)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )

        self.detector = mp_vision.HandLandmarker.create_from_options(options)
        self.tipIds = self.TIP_IDS

        # ---------------------------------------------------------------------
        # Temporal Wrist Anchoring State Machine Parameters (Paper Section III-A)
        # ---------------------------------------------------------------------
        self.target_wrist: Optional[Tuple[float, float]] = None  # Previous wrist anchor w_{t-1}
        self.target_hand = None                                 # Reference active hand object
        self.selected_hand = None                               # Currently selected hand landmarks
        self.lost_frames = 0                                    # Occlusion frame counter
        self.max_lost_frames = 20                               # Hysteresis limit M_lost = 20 frames
        self.max_wrist_distance = 0.25                          # Gating threshold D_max = 0.25 (normalized)

        # Microsecond timestamp tracker for video-mode frame indexing
        self.start_time = time.time()
        self.timestamp_ms = 0

        logger.debug(
            "handDetector initialized (maxHands=%d, detectionCon=%.2f, trackCon=%.2f)",
            maxHands,
            detectionCon,
            trackCon,
        )

    def _wrist_distance(self, hand) -> float:
        """
        Computes Euclidean distance between candidate wrist and prior anchor wrist.
        Implements Equation 1: d(h) = ||w_h - w_{t-1}||_2
        """
        if self.target_wrist is None:
            return float("inf")

        wrist = hand[0]  # Landmark 0 represents the anatomical wrist root

        dx = wrist.x - self.target_wrist[0]
        dy = wrist.y - self.target_wrist[1]

        return math.hypot(dx, dy)

    def _selectTargetHand(self):
        """
        State machine selector enforcing spatial target-hand continuity (Section III-A).
        Selects candidate hand minimizing Euclidean distance subject to d(h) <= D_max.
        """
        if self.results is None:
            return None

        hands = self.results.hand_landmarks

        # Case 1: No hand detected in current frame -> increment occlusion counter
        if not hands:
            if self.target_wrist is not None:
                self.lost_frames += 1
                if self.lost_frames > self.max_lost_frames:
                    # Anchor reset after M_lost consecutive lost frames
                    self.target_wrist = None
                    self.target_hand = None
            return None

        # Case 2: No prior target locked -> latch first detected candidate hand
        if self.target_wrist is None:
            selected = hands[0]
            wrist = selected[0]
            self.target_wrist = (wrist.x, wrist.y)
            self.target_hand = selected
            self.lost_frames = 0
            return selected

        # Case 3: Identify nearest candidate hand matching prior anchor trajectory
        closest_hand = None
        closest_distance = float("inf")

        for hand in hands:
            distance = self._wrist_distance(hand)
            if distance < closest_distance:
                closest_distance = distance
                closest_hand = hand

        # Validate against spatial gating threshold D_max = 0.25 (Equation 1)
        if closest_distance <= self.max_wrist_distance:
            selected = closest_hand
            wrist = selected[0]
            self.target_wrist = (wrist.x, wrist.y)
            self.target_hand = selected
            self.lost_frames = 0
            return selected

        # Distance threshold violated -> retain anchor state within hysteresis buffer
        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.target_wrist = None
            self.target_hand = None
            self.lost_frames = 0

        return None

    def findHands(self, img, draw: bool = True):
        """
        Processes an incoming BGR frame through MediaPipe HandLandmarker.

        Args:
            img: Input image frame in OpenCV BGR format.
            draw: Flag to render skeletal landmarks and connections.

        Returns:
            Processed image frame with optional skeletal annotations.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        self.timestamp_ms = int((time.time() - self.start_time) * 1000)
        self.results = self.detector.detect_for_video(mpImage, self.timestamp_ms)

        # Select anchor hand using temporal continuity algorithm
        self.selected_hand = self._selectTargetHand()

        if self.selected_hand is None:
            return img

        # Optional visualization of 21 landmark points and skeletal topology
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
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
                (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
                (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
                (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
                (0, 17),                               # Palm base
            ]

            for start, end in connections:
                cv2.line(img, points[start], points[end], (0, 255, 0), 2)

        return img

    def findPosition(
        self, img, handNo: int = 0, draw: bool = True
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Extracts 2D pixel coordinates and bounding box for the selected anchor hand.

        Returns:
            lmList: List of landmark items [id, pixel_x, pixel_y].
            bbox: Bounding box coordinates [xmin, ymin, xmax, ymax].
        """
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

    def getHandedness(
        self, handNo: int = 0
    ) -> Optional[str]: 
        """
        Retrieves canonical handedness label ('Left' or 'Right') for the anchor hand.
        """
        if self.selected_hand is None:
            return None

        if self.results and self.results.handedness:
            for i, hand in enumerate(self.results.hand_landmarks):
                if hand is self.selected_hand:
                    return self.results.handedness[i][0].category_name

        return None

    def fingersUp(self) -> List[int]:
        """
        Evaluates binary finger extension state based on fingertip vs joint coordinates.
        """
        fingers: List[int] = []

        if len(self.lmList) == 0:
            return []

        # Thumb extension check (horizontal axis)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Four fingers extension check (vertical axis)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(
        self, p1: int, p2: int, img, draw: bool = True
    ) -> Tuple[float, any, List[int]]:
        """
        Computes Euclidean pixel distance between two specified landmark indices.
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
        return (length, img, [x1, y1, x2, y2, cx, cy])

    def resetTarget(self):
        """
        Explicitly resets the wrist anchor state machine.
        """
        self.target_wrist = None
        self.target_hand = None
        self.selected_hand = None
        self.lost_frames = 0

    def close(self):
        """
        Safely releases underlying MediaPipe Task graph resources.
        """
        if self.detector is not None:
            self.detector.close()