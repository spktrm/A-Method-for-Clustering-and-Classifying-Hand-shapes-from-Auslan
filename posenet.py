import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os 

from alive_progress import alive_bar
from rotate import rotate_3d

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
# file = "Basic words - Auslan.mp4"
# file = "Auslan COVID-19 Update - 29 Jan 2021.mp4"
# file = "WIN_20210418_10_11_51_Pro.mp4"
file = "WIN_20210422_17_16_38_Pro.mp4"

if not os.path.exists(file.split('.')[0] + "/"):
    os.mkdir(file.split('.')[0] + "/")
    os.mkdir(file.split('.')[0] + "/frames/")

cap = cv2.VideoCapture(file)
if file != 0:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

right_hand = []
right_hand_paths = []
left_hand = []
left_hand_paths = []

model = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9, max_num_hands=2)

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

frame_no = 0
with alive_bar(frame_count) as bar:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        file_path = file.split('.')[0] + "/frames/" + str(frame_no) + ".jpg"

        if results.multi_hand_landmarks is not None:
            for hand_landmarks_idx in range(len(results.multi_hand_landmarks)):
                hand_landmarks = results.multi_hand_landmarks[hand_landmarks_idx]

                keypoints = results.multi_hand_landmarks[hand_landmarks_idx].landmark._values
                kp = []
                for point in keypoints:
                    kp.append(np.array([point.x, point.y, point.z]))

                label = results.multi_handedness[0].classification._values[0].label
                kp = np.array(kp)

                if label == 'Right':
                    right_hand.append(kp.flatten())
                    right_hand_paths.append(file_path)
                else:
                    # kp[:, 0] *= -1
                    left_hand.append(kp.flatten())
                    left_hand_paths.append(file_path)

        if not os.path.exists(file_path):
            cv2.imwrite(file_path, image)

        frame_no += 1
        bar()

        # cv2.imshow('MediaPipe Hands', image)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

cap.release()

right_hand_df = pd.DataFrame(np.array(right_hand))
right_hand_paths = pd.DataFrame(right_hand_paths)
right_hand_df.to_csv(file.split('.')[0] + "/right_hand_keypoints.csv")
right_hand_paths.to_csv(file.split('.')[0] + "/right_hand_paths.csv")

left_hand_df = pd.DataFrame(np.array(left_hand))
left_hand_paths = pd.DataFrame(left_hand_paths)
left_hand_df.to_csv(file.split('.')[0] + "/left_hand_keypoints.csv")
left_hand_paths.to_csv(file.split('.')[0] + "/left_hand_paths.csv")
