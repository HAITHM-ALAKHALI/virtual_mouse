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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Utils")))
import HandTrackingModule as htm

logger = logging.getLogger(__name__)

mouse = Controller()


class VirtualMouseController:

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
        self._smoothening = smoothening
        self._deadzone = deadzone
        self._scroll_speed = scroll_speed
        self.cam_width = cam_width
        self.cam_height = cam_height

        self._stop_event = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._current_action: str = ""
        self._current_fps: float = 0.0

        self._clickDelay_raw: int = clickDelay
        self._click_interval: float = max(0.05, clickDelay * 0.025)
        self._right_clicked = False

        self._last_left_click_time: float = 0.0
        self._last_right_click_time: float = 0.0

        self._model = self._load_model()

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
        # التحديث اللحظي للفاصل الزمني بالثواني
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
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def start(self) -> None:
        self._stop_event.clear()
        drag_lost_frames = 0
        scroll_delay = 0

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)

        if not cap.isOpened():
            logger.error("Failed to open camera.")
            raise RuntimeError(
                "Could not open the webcam. Check your camera connection."
            )

        detector = htm.handDetector(maxHands=1)

        root = tk.Tk()
        wScr = root.winfo_screenwidth()
        hScr = root.winfo_screenheight()
        root.destroy()

        plocX, plocY = 0.0, 0.0
        clocX, clocY = 0.0, 0.0
        pTime = 0.0
        dragging = False
        drag_change_frames = 0

        logger.info("Virtual Mouse started — show your hand to the camera.")

        try:
            while not self._stop_event.is_set():
                success, img = cap.read()
                if not success:
                    logger.warning("Camera read failed — retrying …")
                    continue

                img = cv2.flip(img, 1)
                img = detector.findHands(img)
                lmList, bbox = detector.findPosition(img, draw=False)
                handedness = detector.getHandedness()

                if handedness == "Right":
                    actual_hand = "Left"
                elif handedness == "Left":
                    actual_hand = "Right"
                else:
                    actual_hand = None

                if len(lmList) != 0:
                    drag_lost_frames = 0
                    wrist_x, wrist_y = lmList[0][1], lmList[0][2]

                    middle_mcp_x = lmList[9][1] - wrist_x
                    middle_mcp_y = lmList[9][2] - wrist_y

                    scale = np.sqrt(middle_mcp_x**2 + middle_mcp_y**2)

                    if scale == 0:
                        continue

                    hand_data = []
                    for lm in lmList:
                        norm_x = (lm[1] - wrist_x) / scale
                        norm_y = (lm[2] - wrist_y) / scale

                        if actual_hand == "Left":
                            norm_x = -norm_x

                        hand_data.extend([norm_x, norm_y])

                    prediction = self._model.predict([hand_data])[0]
                    self._current_action = prediction

                    x1, y1 = lmList[8][1], lmList[8][2]

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

                    if prediction == "Drag":
                        self._right_clicked = False
                        drag_change_frames = 0
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
                        clocX = plocX + (x3 - plocX) / self._smoothening
                        clocY = plocY + (y3 - plocY) / self._smoothening

                        if not dragging:
                            mouse.press(Button.left)
                            dragging = True

                        mouse.position = (int(clocX), int(clocY))
                        cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    else:
                        if dragging:
                            drag_change_frames += 1
                            if drag_change_frames >= 2:
                                mouse.release(Button.left)
                                dragging = False
                                drag_change_frames = 0
                        else:
                            drag_change_frames = 0

                        if prediction == "Move":
                            self._right_clicked = False
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
                            clocX = plocX + (x3 - plocX) / self._smoothening
                            clocY = plocY + (y3 - plocY) / self._smoothening

                            mouse.position = (int(clocX), int(clocY))
                            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                            plocX, plocY = clocX, clocY

                        elif prediction == "Left_Click":
                            self._right_clicked = False
                            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                            # تثبيت موقع المؤشر أثناء النقر لمنع إلغاء الـ Double Click
                            mouse.position = (int(plocX), int(plocY))

                            if now - self._last_left_click_time >= self._click_interval:
                                mouse.click(Button.left, 1)
                                self._last_left_click_time = now

                        elif prediction == "Right_Click":
                            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                            mouse.position = (int(plocX), int(plocY))

                            if not self._right_clicked:
                                mouse.click(Button.right, 1)
                                self._right_clicked = True

                        elif prediction == "Scroll":
                            self._right_clicked = False
                            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                            centerY = self.cam_height / 2

                            delay_frames = max(0, 6 - self._scroll_speed)
                            steps = max(1, self._scroll_speed - 5)

                            if scroll_delay == 0:
                                if y1 < (centerY - self._deadzone):
                                    mouse.scroll(0, steps)
                                    scroll_delay = delay_frames
                                    cv2.putText(
                                        img,
                                        "Scroll UP",
                                        (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (255, 255, 0),
                                        2,
                                    )
                                elif y1 > (centerY + self._deadzone):
                                    mouse.scroll(0, -steps)
                                    scroll_delay = delay_frames
                                    cv2.putText(
                                        img,
                                        "Scroll DOWN",
                                        (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (255, 255, 0),
                                        2,
                                    )
                            if scroll_delay > 0:
                                scroll_delay -= 1

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
                    self._current_action = ""
                    self._right_clicked = False
                    if dragging:
                        mouse.release(Button.left)
                        dragging = False
                        drag_change_frames = 0

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
            with self._frame_lock:
                self._latest_frame = None
            logger.info("Virtual Mouse stopped.")

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("Stop signal sent.")

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()
