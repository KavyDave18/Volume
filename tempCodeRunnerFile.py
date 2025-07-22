import cv2
import mediapipe as mp
import numpy as np
import math
from pycow.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

cap = VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
draw = mp.solutions.drawing_utils

volControl = AudioUtilities.GetSpeakers().Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(volControl, POINTER(IAudioEndpointVolume))
maxvol , minvol = volume.GetVolumeRange()[:2]
while True:
    ret, img = cap.read() 
    if not ret:
        print("Failed to capture image")
        break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        hands = results.multi_hand_landmarks[0]
        draw.draw_landmarks(img, hands, mpHands.HAND_CONNECTIONS)

        h , w = img.shape[:2]
        lm = hands.landmark
        x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
        x2 , y2 = int(lm[8].x * w), int(lm[8].y * h)
        cx , cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [30, 200], [minvol, maxvol])
        volume.SetMasterVolumeLevel(vol, None)

        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.putText(img, f'Volume: {int(np.interp(length, [30, 200], [0, 100]))}%', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()