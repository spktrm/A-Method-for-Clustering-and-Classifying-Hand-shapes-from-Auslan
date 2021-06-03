import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import os
import time

from rotate import rotate_3d, normalise, subtract_offset

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

sign_names = [
    "STARTING...",
    "ANIMAL",
    "BAD",
    "BENT FLAT",
    "BENT GUN",
    "BENT TWO",
    "CLAW",
    "CLOSED",
    "CUP",
    "EIGHT",
    "ELEVEN",
    "FIST",
    "FIVE",
    "FLAT",
    "FLAT OKAY",
    "FLAT ROUND",
    "FLICK",
    "FOUR",
    "GOOD",
    "GUN",
    "HOOK",
    "I-LOVE-YOU",
    "KEY",
    "LETTER-C",
    "LETTER-M",
    "MIDDLE",
    "OKAY",
    "ONE-HAND-LETTER-D",
    "ONE-HAND-LETTER-K",
    "OPEN SPOON",
    "PLANE",
    "POINT",
    "ROUND",
    "RUDE",
    "SMALL",
    "SPOON",
    "THICK",
    "THREE",
    "TWELVE",
    "TWO",
    "WISH",
    "WRITE"
]

cap = cv2.VideoCapture(0)

start = time.time()
init_timer_val = 10
sign_timer_val = 55
change_time = 5
frame_no = 0

handshapes = "handshapes1/"
hand = []

if not os.path.exists(handshapes):
    os.mkdir(handshapes)

model = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=2)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    scale_percent = 1/(frame.shape[1] / 800)
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    dim = (width, height)
    frame = cv2.resize(frame, dim)

    current_time = time.time()
    time_dif = (current_time - start)
    if time_dif < init_timer_val:
        timer_val = init_timer_val
    else:
        timer_val = sign_timer_val
        
    sign_idx = int((time_dif - init_timer_val) / timer_val)
    count = int((time_dif - init_timer_val) % timer_val) + 1

    try:
        folder_name = str(sign_names[sign_idx])
    except IndexError:
        break

    image = cv2.cvtColor(cv2.flip(frame , 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not os.path.exists(handshapes + folder_name) and folder_name != "STARTING...":
        os.mkdir(handshapes + folder_name)
        frame_no = 0

    if timer_val - count >= change_time and sign_names[sign_idx] != "STARTING...":
        path = handshapes + folder_name + "/" + str(frame_no) + ".jpg"
        cv2.imwrite(path, frame)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks_idx in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[hand_landmarks_idx]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                keypoints = results.multi_hand_landmarks[hand_landmarks_idx].landmark._values
                kp = []
                for point in keypoints:
                    kp.append(np.array([point.x, point.y, point.z]))

                label = results.multi_handedness[0].classification._values[0].label

                keypoints = np.array(kp)
                keypoints = subtract_offset(keypoints)
                keypoints = rotate_3d(keypoints)
                keypoints = normalise(keypoints)

                if label == 'Right':
                    hand.append(np.append(keypoints.flatten(), sign_idx))
                else:
                    keypoints[:, 0] *= -1
                    hand.append(np.append(keypoints.flatten(), sign_idx))

    image = cv2.putText(image, str(timer_val - count), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=4)
    if timer_val - count < change_time:
        try:
            image = cv2.putText(image, "NEXT: {}".format(str(sign_names[sign_idx + 1])), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), thickness=4)
        except IndexError:
            break
    else:
        image = cv2.putText(image, str(sign_names[sign_idx + 1]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), thickness=4)

    # Display the resulting frame
    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break        

    frame_no += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

hand_df = pd.DataFrame(np.array(hand))
hand_df.to_csv(handshapes + "hand_keypoints.csv")