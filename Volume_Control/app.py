import cv2
import mediapipe as mp
import numpy as np
import math
import os

# Initialize camera
cap = cv2.VideoCapture(0)

# MediaPipe hands module setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

# Finger tip landmarks
FINGER_TIPS = [4, 8, 12, 16, 20]

# Volume control constants
min_vol_length = 30
max_vol_length = 200

# Automatically open Spotify (optional)
os.system('open -a Spotify')

# Utility function to detect fingers up
def fingers_up(hand_landmarks):
    fingers = []
    lm = hand_landmarks.landmark
    
    # Thumb
    fingers.append(lm[4].x < lm[3].x)
    
    # Other four fingers
    for id in range(1, 5):
        fingers.append(lm[FINGER_TIPS[id]].y < lm[FINGER_TIPS[id] - 2].y)
    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        draw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        lm = handLms.landmark
        h, w = img.shape[:2]
        x1, y1 = int(lm[4].x * w), int(lm[4].y * h)  # Thumb tip
        x2, y2 = int(lm[8].x * w), int(lm[8].y * h)  # Index tip

        # Calculate distance for volume
        length = math.hypot(x2 - x1, y2 - y1)
        volume_percent = np.interp(length, [min_vol_length, max_vol_length], [0, 100])
        os.system(f"osascript -e 'set volume output volume {int(volume_percent)}'")

        # Gesture detection
        finger_status = fingers_up(handLms)

        # Play/Pause: Fist (all fingers down)
        if finger_status == [False, False, False, False, False]:
            os.system("osascript -e 'tell application \"Spotify\" to playpause'")
            cv2.putText(img, "Play/Pause", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Next: Only Index up
        elif finger_status == [False, True, False, False, False]:
            os.system("osascript -e 'tell application \"Spotify\" to next track'")
            cv2.putText(img, "Next Song", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        # Previous: Only Pinky up
        elif finger_status == [False, False, False, False, True]:
            os.system("osascript -e 'tell application \"Spotify\" to previous track'")
            cv2.putText(img, "Previous Song", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

        # Display volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 2)
        cv2.rectangle(img, (50, int(400 - (volume_percent / 100) * 250)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f'{int(volume_percent)}%', (40, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Spotify Hand Control", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()