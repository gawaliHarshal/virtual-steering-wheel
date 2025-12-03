import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui

# Load steering image
wheel_img = cv2.imread("wheel.png", cv2.IMREAD_UNCHANGED)

def overlay_wheel(frame, angle):
    if wheel_img is None:
        return frame
    h, w, _ = frame.shape
    target_size = int(min(h, w) * 0.7)  # Large overlay
    wheel_resized = cv2.resize(wheel_img, (target_size, target_size))
    wh, ww, _ = wheel_resized.shape
    M = cv2.getRotationMatrix2D((ww // 2, wh // 2), -angle, 1)
    rotated = cv2.warpAffine(wheel_resized, M, (ww, wh), borderMode=cv2.BORDER_TRANSPARENT)
    if rotated.shape[2] == 4:
        alpha = rotated[:, :, 3] / 255.0
        rgb = rotated[:, :, :3]
    else:
        alpha = np.ones((wh, ww))
        rgb = rotated
    x_offset = w // 2 - ww // 2
    y_offset = h // 2 - wh // 2
    for c in range(3):
        frame[y_offset:y_offset+wh, x_offset:x_offset+ww, c] = (
            alpha * rgb[:, :, c] + (1 - alpha) * frame[y_offset:y_offset+wh, x_offset:x_offset+ww, c]
        )
    return frame

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
previous_action = None
accelerating = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    angle = 0

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]
        x1 = int(hand1.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        y1 = int(hand1.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
        x2 = int(hand2.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[1])
        y2 = int(hand2.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(dy, dx))
        distance = math.hypot(dx, dy)

        # Steering
        if angle < -15 and previous_action != "left":
            pyautogui.keyUp("right")
            pyautogui.keyDown("left")
            previous_action = "left"
        elif angle > 15 and previous_action != "right":
            pyautogui.keyUp("left")
            pyautogui.keyDown("right")
            previous_action = "right"
        elif -15 <= angle <= 15 and previous_action is not None:
            pyautogui.keyUp("left")
            pyautogui.keyUp("right")
            previous_action = None

        # Acceleration
        if distance > 200 and not accelerating:
            pyautogui.keyDown("up")
            accelerating = True
        elif distance <= 200 and accelerating:
            pyautogui.keyUp("up")
            accelerating = False
    else:
        if previous_action:
            pyautogui.keyUp("left")
            pyautogui.keyUp("right")
            previous_action = None
        if accelerating:
            pyautogui.keyUp("up")
            accelerating = False

    frame = overlay_wheel(frame, angle)
    cv2.imshow("Virtual Steering", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
