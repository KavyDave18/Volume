import cv2
import mediapipe as mp
import subprocess
import time
import math

def play_spotify():
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to play'])

def pause_spotify():
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to pause'])

def next_spotify():
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to next track'])

def prev_spotify():
    subprocess.run(["osascript", "-e", 'tell application "Spotify" to previous track'])

def set_volume_mac(vol_percent):
    vol_percent = max(0, min(100, vol_percent))
    subprocess.run(["osascript", "-e", f"set volume output volume {int(vol_percent)}"])

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

FINGER_TIPS = [4, 8, 12, 16, 20] 

def fingers_up(hand):
    fingers = []
    fingers.append(hand.landmark[4].x < hand.landmark[3].x)  # Thumb (compare x)
    fingers += [hand.landmark[t].y < hand.landmark[t - 2].y for t in FINGER_TIPS[1:]]
    return fingers

def calc_distance(p1, p2, img_w, img_h):
    x1, y1 = int(p1.x * img_w), int(p1.y * img_h)
    x2, y2 = int(p2.x * img_w), int(p2.y * img_h)
    return math.hypot(x2 - x1, y2 - y1)

cap = cv2.VideoCapture(0)
prev_action = None
last_time = time.time()

min_dist = 20
max_dist = 150
vol_min = 0
vol_max = 100
last_vol_set_time = 0
volume = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            fingers = fingers_up(lm)
            now = time.time()

            if fingers[0] and fingers[1]:
                pinch_distance = calc_distance(lm.landmark[4], lm.landmark[8], w, h)
                volume = ((pinch_distance - min_dist) / (max_dist - min_dist)) * (vol_max - vol_min) + vol_min
                volume = max(vol_min, min(vol_max, volume))
                if now - last_vol_set_time > 0.3:
                    set_volume_mac(volume)
                    last_vol_set_time = now

            if now - last_time > 1:
                if fingers == [False, True, True, False, False] and prev_action != "pause":
                    pause_spotify()
                    prev_action = "pause"
                    last_time = now

                elif fingers == [True, True, True, True, True] and prev_action != "play":
                    play_spotify()
                    prev_action = "play"
                    last_time = now

                elif fingers == [False, True, False, False, False] and prev_action != "next":
                    next_spotify()
                    prev_action = "next"
                    last_time = now

                elif fingers == [False, False, False, False, True] and prev_action != "prev":
                    prev_spotify()
                    prev_action = "prev"
                    last_time = now

    bar_x, bar_y = 50, 100
    bar_width, bar_height = 30, 300
    vol_bar_height = int((volume / 100) * bar_height)
    vol_y = bar_y + bar_height - vol_bar_height

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    cv2.rectangle(frame, (bar_x, vol_y), (bar_x + bar_width, bar_y + bar_height), (0, 255, 0), -1)
    cv2.putText(frame, f'{int(volume)}%', (bar_x - 10, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Spotify Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
