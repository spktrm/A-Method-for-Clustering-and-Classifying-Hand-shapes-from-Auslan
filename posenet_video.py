import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from glob import glob
from rotate import normalise, rotate_3d, subtract_offset, draw_hand_skel_front, draw_hand_skel_side

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

file = "Basic words - Auslan.mp4"
# file = "Auslan COVID-19 Update - 29 Jan 2021.mp4"
# file = "WIN_20210418_10_11_51_Pro.mp4"
# file = "WIN_20210422_17_16_38_Pro.mp4"
cap = cv2.VideoCapture(file)

# cap = cv2.VideoCapture(0)

model_keypoints = mp_hands.Hands(min_detection_confidence=0.55, min_tracking_confidence=0.55, max_num_hands=2)

model_classify = keras.models.load_model("hand_models/modelv4.9")
class_dir = "handshapes/*/"
class_names = names = [os.path.basename(x[0:-1]) for x in glob(class_dir)]

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    scale_percent = 1/(image.shape[1] / 800)
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    image = cv2.resize(cv2.flip(image, 1), dim)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model_keypoints.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks is not None:
        for hand_landmarks_idx in range(len(results.multi_hand_landmarks)):
            hand_landmarks = results.multi_hand_landmarks[hand_landmarks_idx]
            # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = results.multi_hand_landmarks[hand_landmarks_idx].landmark._values
            kp = []
            ratio = image.shape[1] / image.shape[0]
            for point in keypoints:
                kp.append(np.array([point.x * ratio, point.y, point.z * ratio]))

            label = results.multi_handedness[hand_landmarks_idx].classification._values[0].label

            keypoints = np.array(kp)
            keypoints = subtract_offset(keypoints)
            keypoints = rotate_3d(keypoints)
            keypoints = normalise(keypoints)

            if label == "Left":
                mask_left_front = np.zeros((224, 224, 3), dtype='uint8')
                left_front = draw_hand_skel_front(keypoints, mask_left_front, ratio)
                cv2.imshow("Left Front", left_front)

                mask_left_side = np.zeros((224, 224, 3), dtype='uint8')
                left_side = draw_hand_skel_side(keypoints, mask_left_side, ratio)
                cv2.imshow("Left Side", left_side)

                keypoints[:, 0] *= -1

                keypoints = tf.expand_dims(keypoints.flatten(), 0)
                predictions = model_classify.predict(keypoints)
                score = predictions[0]

                class_label = class_names[np.argmax(score)]
                class_score = str(round(100 * np.max(score), 2))

                textsize1 = cv2.getTextSize("Left Hand", cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0]
                textsize2 = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0]
                textsize3 = cv2.getTextSize(class_score, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0]

                start = (0, 0)
                end = (int(max(textsize1[0], textsize2[0], textsize3[0])), 75)
                cv2.rectangle(image, start, end, (0, 0, 0), -1)

                sprite = cv2.imread("handshape_sprites/" + class_label + ".jpg")
                dim = 75
                image[80:80 + dim, 0:0 + dim] = cv2.resize(sprite, (dim, dim))

                cv2.putText(image, "Left Hand", (0, 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, class_label, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, class_score, (0, 75), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

            else:

                mask_right_front = np.zeros((224, 224, 3), dtype='uint8')
                right_front = draw_hand_skel_front(keypoints, mask_right_front, ratio)
                cv2.imshow("Right Front", right_front)

                mask_right_side = np.zeros((224, 224, 3), dtype='uint8')
                right_side = draw_hand_skel_side(keypoints, mask_right_side, ratio)
                cv2.imshow("Right Side", right_side)

                keypoints = tf.expand_dims(keypoints.flatten(), 0)
                predictions = model_classify.predict(keypoints)
                score = predictions[0]

                class_label = class_names[np.argmax(score)]
                class_score = str(round(100 * np.max(score), 2))

                textsize1 = cv2.getTextSize("Right Hand", cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0]
                textsize2 = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0]
                textsize3 = cv2.getTextSize(class_score, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0]

                start = (int(image.shape[1] - max(textsize1[0], textsize2[0], textsize3[0])), 0)
                end = (image.shape[1], 75)
                cv2.rectangle(image, start, end, (0, 0, 0), -1)

                sprite = cv2.imread("handshape_sprites/" + class_label + ".jpg")
                dim = 75
                image[80:80 + dim, int(image.shape[1] - dim):int(image.shape[1])] = cv2.resize(sprite, (dim, dim))

                cv2.putText(image, "Right Hand", (int(image.shape[1] - textsize1[0]), 25), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, class_label, (int(image.shape[1] - textsize2[0]), 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, class_score, (int(image.shape[1] - textsize3[0]), 75), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
