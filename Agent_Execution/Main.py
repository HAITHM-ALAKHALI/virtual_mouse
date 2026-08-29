import cv2
import numpy as np
import pyautogui
import pickle
import os
import sys
import time
import logging
import threading
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Utils")))
import HandTrackingModule as htm

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = False


class VirtualMouseController:

    FRAME_LEFT = 100
    FRAME_TOP = 100
    FRAME_RIGHT = 100
    FRAME_BOTTOM = 150

    def __init__(
        self,
        smoothening: int = 7,
        deadzone: int = 60,
        scroll_speed: int = 50,
        cam_width: int = 700,
        cam_height: int = 500,
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

        self._model = self._load_model()

    @property
    def smoothening(self) -> int:
        return self._smoothening

    @smoothening.setter
    def smoothening(self, value: int) -> None:
        self._smoothening = max(1, int(value))

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
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(
            current_dir, "..", "Model_Training", "virtual_mouse_brain.pkl"
        )

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

        cap = cv2.VideoCapture(0)
        cap.set(3, self.cam_width)
        cap.set(4, self.cam_height)

        if not cap.isOpened():
            logger.error("Failed to open camera.")
            raise RuntimeError(
                "Could not open the webcam. Check your camera connection."
            )

        detector = htm.handDetector(maxHands=1)
        wScr, hScr = pyautogui.size()

        plocX, plocY = 0.0, 0.0
        clocX, clocY = 0.0, 0.0
        pTime = 0.0
        clickDelay = 0
        dragging = False

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

                if len(lmList) != 0:
                    wrist_x, wrist_y = lmList[0][1], lmList[0][2]
                    hand_data = []
                    for lm in lmList:
                        hand_data.extend([lm[1] - wrist_x, lm[2] - wrist_y])

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

                    if prediction == "Move":
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
                        pyautogui.moveTo(clocX, clocY)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    elif prediction == "Left_Click":
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        if clickDelay == 0:
                            pyautogui.click(button="left")
                            clickDelay = 5

                    elif prediction == "Right_Click":
                        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                        if clickDelay == 0:
                            pyautogui.click(button="right")
                            clickDelay = 5

                    elif prediction == "Scroll":
                        cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
                        centerY = self.cam_height / 2

                        if y1 < (centerY - self._deadzone):
                            pyautogui.scroll(self._scroll_speed)
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
                            pyautogui.scroll(-self._scroll_speed)
                            cv2.putText(
                                img,
                                "Scroll DOWN",
                                (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 0),
                                2,
                            )

                    elif prediction == "Drag":
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
                            pyautogui.mouseDown(button="left")
                            dragging = True

                        pyautogui.moveTo(clocX, clocY)
                        cv2.circle(img, (x1, y1), 15, (0, 255, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

                    if dragging and prediction != "Drag":
                        pyautogui.mouseUp(button="left")
                        dragging = False

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

                if clickDelay > 0:
                    clickDelay -= 1

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
                pyautogui.mouseUp(button="left")
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