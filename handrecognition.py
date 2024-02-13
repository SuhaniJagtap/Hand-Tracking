import cv2
import mediapipe as mp
import time

mphands = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils
hands = mphands.Hands()

ctime = 0
ptime = 0

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)
            finger_count = 0
            
            if hand_landmarks.landmark[mphands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mphands.HandLandmark.THUMB_IP].x:
                finger_count += 1
            
            if hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_MCP].y:
                finger_count += 1
            
            if hand_landmarks.landmark[mphands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mphands.HandLandmark.MIDDLE_FINGER_MCP].y:
                finger_count += 1
           
            if hand_landmarks.landmark[mphands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mphands.HandLandmark.RING_FINGER_MCP].y:
                finger_count += 1
            
            if hand_landmarks.landmark[mphands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mphands.HandLandmark.PINKY_MCP].y:
                finger_count += 1

            cv2.putText(img, f'Fingers: {finger_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Tracker", img)
    if cv2.waitKey(1) & 0xFF == ord('f'):
        break

cap.release()
cv2.destroyAllWindows()
