import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os 

from alive_progress import alive_bar
from rotate import normalise, rotate_3d, subtract_offset

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

frame_count = sum(len(files) for _, _, files in os.walk("handshapes"))

hand = []

model = mp_hands.Hands(min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=2)

labels = os.listdir("handshapes")
label_idx = -1
n = 0
with alive_bar(frame_count - 1) as bar:
    for root, dirs, files in os.walk("handshapes"):
        for f in files:
            if f.endswith(".jpg"):
                image = cv2.imread(os.path.join(root, f))

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = model.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks_idx in range(len(results.multi_hand_landmarks)):
                        hand_landmarks = results.multi_hand_landmarks[hand_landmarks_idx]

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
                            hand.append(np.append(keypoints.flatten(), label_idx))
                            n += 1
                        else:
                            keypoints[:, 0] *= -1
                            hand.append(np.append(keypoints.flatten(), label_idx))
                            n += 1
                
                bar()

        label_idx += 1

print("\nAverage {} hands/frame\n".format(round(n/frame_count, 2)))

hand_df = pd.DataFrame(np.array(hand))
hand_df.to_csv("handshapes/hand_keypoints.csv")