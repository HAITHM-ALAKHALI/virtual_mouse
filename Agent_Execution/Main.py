"""
================================================================================
Main.py — VirtualMouseController (Daemon Engine & Action Dispatcher)
Virtual Mouse Control System
Paper Reference: Section III-B ("Bifurcated Feature Normalization Pipeline")
                 Section III-C ("Rule-Based Action Dispatching Engine")
                 Section III-D ("Thread Concurrency and Telemetry Logging")
                 Section IV-E ("Exploratory Live Ablation Analysis", Table VI)
                 Section IV-F ("Operational Consistency and Reliability Audits", Table VII)
                 Section IV-G ("Per-Subject Interactive Audit Traces", Table VIII)
================================================================================

Description:
    Core vision-to-OS-event pipeline executing within an isolated daemon thread.
    Decouples continuous spatial kinematics from rule-based action dispatching
    to guarantee deterministic OS mouse event synthesis on desktop platforms.

Core Architectural Components:
    1. Continuous Kinematics (Eq. 2–4):
       Maps index fingertip landmark (p8) to OS display coordinates with
       exponential smoothing (smoothing divisor alpha in [1, 15]).
    2. 42D Feature Normalization (Eq. 5–8):
       Translates landmarks to wrist origin (p0), normalizes uniform scale
       against palm length ||p9 - p0||_2, and canonicalizes left hands via reflection.
    3. Rule-Based Dispatcher Controls (Section III-C):
       - Wall-Clock Rate Limiting: T_interval = max(0.05, clickDelay * 0.025) s.
       - Actuation Coordinate Freezing: Locks pointer at (plocX, plocY) during clicks.
       - Latched Context-Menu State Machine (L_R): Retains active right-click state.
       - Two-Frame Drag Hysteresis (Delta_f >= 2): Filters transient tracking flicker.
       - Center Scroll Deadzone: Isolates intentional vertical scrolling gestures.
    4. Thread Safety & Live Telemetry (Section III-D, IV-F, IV-G):
       Guards shared frame buffers and circular audit queues via threading mutexes
       (_frame_lock and _log_lock) for millisecond-resolution telemetry logging,
       backing both the 456-entry audit (Table VII) and session snapshots (Table VIII).
================================================================================
"""

import cv2
import numpy as np
from pynput.mouse import Controller, Button
import pickle
import os
import sys
import time
import logging
import threading
from typing import Optional
import tkinter as tk
from collections import deque
from datetime import datetime

# -----------------------------------------------------------------------------
# Telemetry Output Destination & Module Path Resolution
# -----------------------------------------------------------------------------
AUDIT_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mouse_audit.txt"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Utils")))
import HandTrackingModule as htm

logger = logging.getLogger(__name__)

# Native OS input injection controller interface (pynput)
mouse = Controller()


class VirtualMouseController:
    """
    Asynchronous daemon controller orchestrating computer vision hand tracking,
    42D feature normalization, SVM inference, and rule-based OS event dispatching.
    """

    # Internal interaction bounding box margins (in camera pixels)
    FRAME_LEFT = 100
    FRAME_TOP = 100
    FRAME_RIGHT = 100
    FRAME_BOTTOM = 100

    def __init__(
        self,
        smoothening: int = 7,
        deadzone: int = 60,
        scroll_speed: int = 3,
        cam_width: int = 640,
        cam_height: int = 480,
        clickDelay: int = 10,
    ) -> None:
        """
        Initializes controller parameters, synchronization locks, and telemetry buffers.
        """
        self._smoothening = smoothening
        self._deadzone = deadzone
        self._scroll_speed = scroll_speed
        self.cam_width = cam_width
        self.cam_height = cam_height

        self._scroll_delay = 0

        # Concurrency primitives
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()  # Guards GUI preview frame sharing
        self._current_action: str = ""
        self._current_fps: float = 0.0

        # ---------------------------------------------------------------------
        # Action Dispatcher State Parameters (Section III-C)
        # ---------------------------------------------------------------------
        self._clickDelay_raw: int = clickDelay
        # Wall-clock refractory gate: T_interval = max(0.05, clickDelay * 0.025) s
        self._click_interval: float = max(0.05, clickDelay * 0.025)
        self._right_clicked = False  # Context-menu retention latch boolean (L_R)
        self._last_left_click_time: float = 0.0
        self._last_right_click_time: float = 0.0

        # Telemetry circular and full-session buffers (Section III-D)
        self._event_log: deque = deque(maxlen=20)
        self._log_lock = threading.Lock()
        self._log_counter: int = 0
        self._full_session_log: list = []

        # Load serialized SVM classifier pipeline (virtual_mouse_brain.pkl)
        self._model = self._load_model()

    # -------------------------------------------------------------------------
    # Thread-Safe Property Accessors & Mutators
    # -------------------------------------------------------------------------
    @property
    def smoothening(self) -> int:
        return self._smoothening

    @smoothening.setter
    def smoothening(self, value: int) -> None:
        self._smoothening = max(1, int(value))

    @property
    def clickDelay(self) -> int:
        return self._clickDelay_raw

    @clickDelay.setter
    def clickDelay(self, value: int) -> None:
        self._clickDelay_raw = max(0, int(value))
        self._click_interval = max(0.05, self._clickDelay_raw * 0.025)

    @property
    def deadzone(self) -> int:
        return self._deadzone

    @deadzone.setter
    def deadzone(self, value: int) -> None:
        self._deadzone = max(0, int(value))

    @property
    def scroll_speed(self) -> int:
        return self._scroll_speed

    @scroll_speed.setter
    def scroll_speed(self, value: int) -> None:
        self._scroll_speed = max(1, int(value))

    @property
    def current_action(self) -> str:
        return self._current_action

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @staticmethod
    def _load_model():
        """
        Deserializes the pre-trained SVM model pipeline (StandardScaler + SVC RBF).
        Handles relative paths across development and PyInstaller bundled runtimes.
        """
        if getattr(sys, "frozen", False):
            base_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
            model_path = os.path.join(base_dir, "virtual_mouse_brain.pkl")
        else:
            current_dir = os.path.dirname(__file__)
            model_path = os.path.join(
                current_dir, "..", "Model_Training", "virtual_mouse_brain.pkl"
            )

        if not os.path.isfile(model_path):
            alt_path = os.path.join(
                os.path.dirname(__file__), "virtual_mouse_brain.pkl"
            )
            if os.path.isfile(alt_path):
                model_path = alt_path

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}. "
                "Please run model training first."
            )

        logger.info("Loading Virtual Mouse Brain from %s …", model_path)
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
        return model

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Returns a thread-safe copy of the latest processed video frame for GUI preview.
        """
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def log_event(self, action: str, details: str = "") -> None:
        """
        Appends millisecond-stamped interaction entries to telemetry buffers (Section III-D).
        Logs both dispatched OS actions and suppressed candidate states.
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{timestamp}] {action:<22} | {details}"
        with self._log_lock:
            self._event_log.append(entry)
            self._full_session_log.append(entry)
            self._log_counter += 1

    @property
    def log_counter(self) -> int:
        return self._log_counter

    def get_latest_events(self) -> list:
        """
        Fetches snapshot of circular preview queue for GUI log panel (deque maxlen=20).
        """
        with self._log_lock:
            return list(self._event_log)

    def get_all_session_events(self) -> list:
        """
        Fetches full-session telemetry array for export to mouse_audit.txt.
        """
        with self._log_lock:
            return list(self._full_session_log)

    def start(self) -> None:
        """
        Main execution loop executing on the dedicated background daemon thread.
        Orchestrates frame capture, landmark extraction, normalization, SVM inference,
        action dispatching, and OS event injection.
        """
        self._stop_event.clear()

        with self._log_lock:
            self._full_session_log.clear()
            self._event_log.clear()

        # Initialize hardware capture interface with DirectShow backend for Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        if not cap.isOpened():
            logger.error("Failed to open camera.")
            raise RuntimeError(
                "Could not open the webcam. Check your camera connection."
            )

        # =====================================================================
        # ABLATION BENCHMARK FLAGS (Paper Section IV-E & Table VI)
        # Toggle flags to replicate exploratory live ablation conditions.
        # =====================================================================
        ENABLE_COORD_FREEZE = True  # Lock pointer at (plocX, plocY) during clicks
        ENABLE_TIME_DEBOUNCE = True  # Wall-clock left-click refractory gate (Delta_t)
        ENABLE_MENU_LATCH = True  # Latched context-menu state machine (L_R)
        ENABLE_DRAG_HYSTERESIS = True  # Two-frame persistence filter (Delta_f >= 2)
        ENABLE_SCROLL_DEADZONE = True  # Vertical center deadband (|Delta_Y| > deadzone)
        ENABLE_SMOOTHING = True  # Exponential moving average filter (alpha)

        # Instantiate MediaPipe HandLandmarker wrapper with temporal anchoring
        detector = htm.handDetector(maxHands=1)

        # Inquire native desktop resolution
        root = tk.Tk()
        wScr = root.winfo_screenwidth()
        hScr = root.winfo_screenheight()
        root.destroy()

        plocX, plocY = 0.0, 0.0  # Preceding cursor coordinates (p_{locX}, p_{locY})
        clocX, clocY = 0.0, 0.0  # Current cursor coordinates (c_{locX}, c_{locY})
        pTime = 0.0  # Prior frame timestamp for FPS profiling
        dragging = False  # Active drag transaction state flag
        drag_change_frames = 0  # Two-frame hysteresis counter (Delta_f)

        logger.info("Virtual Mouse started — show your hand to the camera.")

        try:
            while not self._stop_event.is_set():
                success, img = cap.read()
                if not success:
                    logger.warning("Camera read failed — retrying …")
                    continue

                # Mirror image horizontally to provide natural interaction feedback
                img = cv2.flip(img, 1)

                # Perception stage: extract 21 3D landmarks via MediaPipe Tasks
                img = detector.findHands(img)
                lmList, bbox = detector.findPosition(img, draw=False)
                handedness = detector.getHandedness()

                # Adjust for horizontal mirror inversion to obtain true anatomical handedness
                if handedness == "Right":
                    actual_hand = "Left"
                elif handedness == "Left":
                    actual_hand = "Right"
                else:
                    actual_hand = None

                if len(lmList) != 0:
                    # ---------------------------------------------------------
                    # 42D Feature Normalization Pipeline (Paper Section III-B)
                    # ---------------------------------------------------------
                    # 1. Translation Normalization: Wrist p0 as origin (Eq. 5)
                    wrist_x, wrist_y = lmList[0][1], lmList[0][2]

                    # 2. Anatomical Palm Scale Factor: ||p9 - p0||_2 (Eq. 6)
                    middle_mcp_x = lmList[9][1] - wrist_x
                    middle_mcp_y = lmList[9][2] - wrist_y
                    scale = np.sqrt(middle_mcp_x**2 + middle_mcp_y**2)

                    if scale == 0:
                        continue

                    # 3. Uniform Scaling & Handedness Reflection (Eq. 7 & 8)
                    hand_data = []
                    for lm in lmList:
                        norm_x = (lm[1] - wrist_x) / scale
                        norm_y = (lm[2] - wrist_y) / scale

                        # Canonicalize left hand into unified right-hand representation
                        if actual_hand == "Left":
                            norm_x = -norm_x

                        hand_data.extend([norm_x, norm_y])

                    # ---------------------------------------------------------
                    # Statistical Classification Inference (Paper Section IV-B)
                    # ---------------------------------------------------------
                    prediction = self._model.predict([hand_data])[0]
                    self._current_action = prediction

                    # Index fingertip landmark (p8) for continuous cursor kinematics
                    x1, y1 = lmList[8][1], lmList[8][2]

                    # Render active interaction boundary box
                    cv2.rectangle(
                        img,
                        (self.FRAME_LEFT, self.FRAME_TOP),
                        (
                            self.cam_width - self.FRAME_RIGHT,
                            self.cam_height - self.FRAME_BOTTOM,
                        ),
                        (255, 0, 255),
                        2,
                    )

                    now = time.time()

                    # =========================================================================
                    # 1. (Drag State Machine & Kinematics - Section III-C)
                    # =========================================================================
                    if prediction == "Drag":
                        self._right_clicked = False
                        drag_change_frames = 0

                        # Map continuous index tip coordinates to display resolution (Eq. 2 & 3)
                        x3 = np.interp(
                            x1,
                            (self.FRAME_LEFT, self.cam_width - self.FRAME_RIGHT),
                            (0, wScr),
                        )
                        y3 = np.interp(
                            y1,
                            (self.FRAME_TOP, self.cam_height - self.FRAME_BOTTOM),
                            (0, hScr),
                        )

                        # Exponential smoothing filter (Equation 4)
                        if ENABLE_SMOOTHING:
                            clocX = plocX + (x3 - plocX) / self._smoothening
                            clocY = plocY + (y3 - plocY) / self._smoothening
                        else:
                            clocX, clocY = x3, y3

                        # Dispatch OS drag initiation event
                        if not dragging:
                            mouse.press(Button.left)
                            dragging = True
                            self.log_event(
                                "DRAG_START", f"Anchor: ({int(clocX)}, {int(clocY)})"
                            )

                        # Continuous cursor trajectory update
                        mouse.position = (int(clocX), int(clocY))
                        cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    else:
                        # =====================================================================
                        # 2. (Drag Release via Hysteresis)
                        # =====================================================================
                        if dragging:
                            if ENABLE_DRAG_HYSTERESIS:
                                # Require two consecutive non-drag frames (Delta_f >= 2)
                                drag_change_frames += 1
                                if drag_change_frames >= 2:
                                    mouse.release(Button.left)
                                    dragging = False
                                    drag_change_frames = 0
                                    self.log_event(
                                        "DRAG_DROP",
                                        f"Released at: ({int(clocX)}, {int(clocY)})",
                                    )
                            else:
                                # Baseline: Immediate release on single-frame drop (premature drop)
                                mouse.release(Button.left)
                                dragging = False
                                self.log_event(
                                    "DRAG_DROP_PREMATURE",
                                    f"Pos: ({int(clocX)}, {int(clocY)})",
                                )
                        else:
                            drag_change_frames = 0

                        # =====================================================================
                        # 3.(Primary Discrete Action Dispatcher)
                        # =====================================================================
                        if prediction == "Move":
                            self._right_clicked = False

                            # Interpolate continuous coordinates to display bounds (Eq. 2 & 3)
                            x3 = np.interp(
                                x1,
                                (self.FRAME_LEFT, self.cam_width - self.FRAME_RIGHT),
                                (0, wScr),
                            )
                            y3 = np.interp(
                                y1,
                                (self.FRAME_TOP, self.cam_height - self.FRAME_BOTTOM),
                                (0, hScr),
                            )

                            # Exponential smoothing filter (Equation 4)
                            if ENABLE_SMOOTHING:
                                clocX = plocX + (x3 - plocX) / self._smoothening
                                clocY = plocY + (y3 - plocY) / self._smoothening
                            else:
                                clocX, clocY = x3, y3

                            mouse.position = (int(clocX), int(clocY))
                            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                            plocX, plocY = clocX, clocY

                        # Guard: Dispatch clicks and scrolling only when not dragging
                        elif not dragging:
                            # -------------------------------------------------
                            # Left Click Action Subsystem (Section III-C)
                            # -------------------------------------------------
                            if prediction == "Left_Click":
                                self._right_clicked = False
                                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

                                # Actuation Coordinate Freezing: lock to pre-actuation anchor
                                if ENABLE_COORD_FREEZE:
                                    mouse.position = (int(plocX), int(plocY))
                                else:
                                    x3 = np.interp(
                                        x1,
                                        (
                                            self.FRAME_LEFT,
                                            self.cam_width - self.FRAME_RIGHT,
                                        ),
                                        (0, wScr),
                                    )
                                    y3 = np.interp(
                                        y1,
                                        (
                                            self.FRAME_TOP,
                                            self.cam_height - self.FRAME_BOTTOM,
                                        ),
                                        (0, hScr),
                                    )
                                    mouse.position = (int(x3), int(y3))

                                time_diff = now - self._last_left_click_time

                                # Wall-Clock Rate Limiting Refractory Gate (Delta_t >= T_interval)
                                if ENABLE_TIME_DEBOUNCE:
                                    if time_diff >= self._click_interval:
                                        mouse.click(Button.left, 1)
                                        self._last_left_click_time = now
                                        current_pos = mouse.position
                                        self.log_event(
                                            "LEFT_CLICK",
                                            f"Pos: ({int(current_pos[0])}, {int(current_pos[1])}) | Debounce: {self._click_interval:.2f}s",
                                        )
                                    else:
                                        remaining = self._click_interval - time_diff
                                        self.log_event(
                                            "SUPPRESSED",
                                            f"LEFT_CLICK cooldown | Held: {remaining:.3f}s remaining",
                                        )
                                else:
                                    # Baseline: Unconstrained dispatch resulting in click cascades
                                    mouse.click(Button.left)
                                    current_pos = mouse.position
                                    self.log_event(
                                        "LEFT_CLICK_CASCADE",
                                        f"Pos: ({int(current_pos[0])}, {int(current_pos[1])})",
                                    )

                            # -------------------------------------------------
                            # Right Click Context-Menu Subsystem (Equation 9)
                            # -------------------------------------------------
                            elif prediction == "Right_Click":
                                cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)

                                # Actuation Coordinate Freezing
                                if ENABLE_COORD_FREEZE:
                                    mouse.position = (int(plocX), int(plocY))
                                else:
                                    x3 = np.interp(
                                        x1,
                                        (
                                            self.FRAME_LEFT,
                                            self.cam_width - self.FRAME_RIGHT,
                                        ),
                                        (0, wScr),
                                    )
                                    y3 = np.interp(
                                        y1,
                                        (
                                            self.FRAME_TOP,
                                            self.cam_height - self.FRAME_BOTTOM,
                                        ),
                                        (0, hScr),
                                    )
                                    mouse.position = (int(x3), int(y3))

                                current_pos = mouse.position

                                # Latched Context-Menu State Machine (L_R)
                                if ENABLE_MENU_LATCH:
                                    if not self._right_clicked:
                                        mouse.click(Button.right, 1)
                                        self._right_clicked = True
                                        self.log_event(
                                            "RIGHT_CLICK",
                                            f"Pos: ({int(current_pos[0])}, {int(current_pos[1])}) | Latched",
                                        )
                                    else:
                                        self.log_event(
                                            "SUPPRESSED",
                                            f"RIGHT_CLICK held | Context menu retained at ({int(current_pos[0])}, {int(current_pos[1])})",
                                        )
                                else:
                                    # Baseline: Continuous triggers dismissing context menus
                                    mouse.click(Button.right, 1)
                                    self.log_event(
                                        "RIGHT_CLICK_REDUNDANT",
                                        f"Pos: ({int(current_pos[0])}, {int(current_pos[1])})",
                                    )

                            # -------------------------------------------------
                            # Scroll Subsystem with Center Deadzone (Section III-C)
                            # -------------------------------------------------
                            elif prediction == "Scroll":
                                self._right_clicked = False
                                cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                                centerY = self.cam_height / 2

                                delay_frames = max(0, 6 - self._scroll_speed)
                                steps = max(1, self._scroll_speed - 5)

                                delta_y = centerY - y1
                                abs_delta = abs(delta_y)

                                if self._scroll_delay == 0:
                                    # Center deadzone evaluation (|Delta_Y| > deadzone)
                                    is_outside_deadzone = (
                                        abs_delta > self._deadzone
                                        if ENABLE_SCROLL_DEADZONE
                                        else True
                                    )

                                    if is_outside_deadzone:
                                        if delta_y > 0:
                                            mouse.scroll(0, steps)
                                            self._scroll_delay = delay_frames
                                            cv2.putText(
                                                img,
                                                "Scroll UP",
                                                (50, 100),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (255, 255, 0),
                                                2,
                                            )
                                            self.log_event(
                                                "SCROLL_UP",
                                                f"Steps: {steps} | ΔY: {int(delta_y)}",
                                            )
                                        elif delta_y < 0:
                                            mouse.scroll(0, -steps)
                                            self._scroll_delay = delay_frames
                                            cv2.putText(
                                                img,
                                                "Scroll DOWN",
                                                (50, 150),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (255, 255, 0),
                                                2,
                                            )
                                            self.log_event(
                                                "SCROLL_DOWN",
                                                f"Steps: {steps} | ΔY: {int(-delta_y)}",
                                            )
                                    else:
                                        self.log_event(
                                            "SUPPRESSED",
                                            f"SCROLL within deadzone | ΔY: {int(delta_y)} (Threshold: {self._deadzone})",
                                        )

                                if self._scroll_delay > 0:
                                    self._scroll_delay -= 1

                    # Overlay active predicted action label
                    cv2.putText(
                        img,
                        f"Action: {prediction}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # Hand tracking lost: unlock context menu latch and release drag
                    self._current_action = ""
                    self._right_clicked = False
                    if dragging:
                        mouse.release(Button.left)
                        dragging = False
                        drag_change_frames = 0

                # Compute daemon loop execution frame rate
                cTime = time.time()
                if pTime != 0:
                    self._current_fps = 1.0 / (cTime - pTime)
                else:
                    self._current_fps = 0.0
                pTime = cTime

                cv2.putText(
                    img,
                    f"FPS: {self._current_fps:.1f}",
                    (self.cam_width - 200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Thread-safe frame copy export to GUI thread
                with self._frame_lock:
                    self._latest_frame = img

        except Exception:
            logger.exception("Unexpected error in Virtual Mouse loop")
        finally:
            if dragging:
                mouse.release(Button.left)
            cap.release()
            cv2.destroyAllWindows()
            self._current_action = ""
            self._current_fps = 0.0
            self._right_clicked = False
            with self._frame_lock:
                self._latest_frame = None
            logger.info("Virtual Mouse stopped.")

    def stop(self) -> None:
        """
        Signals daemon execution loop to terminate safely.
        """
        self._stop_event.set()
        logger.info("Stop signal sent.")

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()
