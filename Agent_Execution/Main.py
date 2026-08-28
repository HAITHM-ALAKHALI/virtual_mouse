import cv2
import numpy as np
import pyautogui
import pickle
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Utils")))
import HandTrackingModule as htm

current_dir = os.path.dirname(__file__)
model_path = os.path.join(
    current_dir, "..", "Model_Training", "virtual_mouse_brain.pkl"
)

print("Loading Virtual Mouse Brain...")
with open(model_path, "rb") as f:
    model = pickle.load(f)

wCam, hCam = 700, 500
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()

frameLeft = 100
frameTop = 100
frameRight = 100
frameBottom = 150

smoothening = 7
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cTime = 0
pTime = 0

clickDelay = 0
dragging = False

print("System Ready! Show your hand to the camera.")

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        wrist_x, wrist_y = lmList[0][1], lmList[0][2]
        hand_data = []
        for lm in lmList:
            norm_x = lm[1] - wrist_x
            norm_y = lm[2] - wrist_y
            hand_data.extend([norm_x, norm_y])

        prediction = model.predict([hand_data])[0]

        x1, y1 = lmList[8][1], lmList[8][2]
        cv2.rectangle(
            img,
            (frameLeft, frameTop),
            (wCam - frameRight, hCam - frameBottom),
            (255, 0, 255),
            2,
        )

        # 4. تنفيذ الأوامر بناءً على قرار الذكاء الاصطناعي

        # -- وضع التحريك --
        if prediction == "Move":
            x3 = np.interp(x1, (frameLeft, wCam - frameRight), (0, wScr))
            y3 = np.interp(y1, (frameTop, hCam - frameBottom), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            pyautogui.moveTo(clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        elif prediction == "Left_Click":
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            if clickDelay == 0:
                pyautogui.click(button="left")
                clickDelay = 10

        elif prediction == "Right_Click":
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            if clickDelay == 0:
                pyautogui.click(button="right")
                clickDelay = 10

        elif prediction == "Scroll":
            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)

            centerY = hCam / 2
            deadzone = 60

            if y1 < (centerY - deadzone):
                pyautogui.scroll(50)
                cv2.putText(
                    img,
                    "Scroll UP",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

            elif y1 > (centerY + deadzone):
                pyautogui.scroll(-50)
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
            x3 = np.interp(x1, (frameLeft, wCam - frameRight), (0, wScr))
            y3 = np.interp(y1, (frameTop, hCam - frameBottom), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

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

    if clickDelay > 0:
        clickDelay -= 1

    cTime = time.time()

    if pTime != 0:
        fps = 1 / (cTime - pTime)
    else:
        fps = 0

    pTime = cTime
    pTime = cTime
    cv2.putText(
        img,
        f"FPS: {fps:.1f}",
        (wCam - 120, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("AI Virtual Mouse V1", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
