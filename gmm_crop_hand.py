import cv2
import numpy as np
import pandas as pd
import os 

from alive_progress import alive_bar
from rotate import rotate_3d

# For webcam input:
# file = "Basic words - Auslan"
# file = "Auslan COVID-19 Update - 29 Jan 2021"
# file = "WIN_20210418_10_11_51_Pro"
file = "WIN_20210422_17_16_38_Pro"

file_count = sum(len(files) for _, _, files in os.walk(file + "/gmm_right_hand_clusters"))

print("loading hand csv file...")
df = pd.read_csv(file + "/right_hand_keypoints.csv")
df = np.array(df)
df = df[:, 1:]
paths = pd.read_csv(file + "/right_hand_paths.csv")
paths = np.array(paths)
paths = paths[:, 1:]
print("hand csv file loaded!\n")

frame_no = 0
with alive_bar(file_count) as bar:
    for root, dirs, files in os.walk(file + "/gmm_right_hand_clusters"):
        for f in files:
            if f.endswith(".jpg"):
                image = cv2.imread(os.path.join(root, f))
                keypoints = df[int(f.split('.')[0])]

                x1 = np.min(keypoints[0::3]) * image.shape[1]
                x2 = np.max(keypoints[0::3]) * image.shape[1]
                y1 = np.min(keypoints[1::3]) * image.shape[0]
                y2 = np.max(keypoints[1::3]) * image.shape[0]

                if x2 - x1 >= y2 - y1:
                    d = (x2 - x1) * 0.1
                else:
                    d = (y2 - y1) * 0.1

                x1 -= d
                y1 -= d
                x2 += d
                y2 += d

                if x2 - x1 >= y2 - y1:
                    y1 -= ((x2 - x1) - (y2 - y1))/2
                    y2 += ((x2 - x1) - (y2 - y1))/2
                else:
                    x1 -= ((y2 - y1) - (x2 - x1))/2
                    x2 += ((y2 - y1) - (x2 - x1))/2
                
                if x1 < 0:
                    x1 = 0
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                if y1 < 0:
                    y1 = 0
                if y2 > image.shape[0]:
                    y2 = image.shape[0]

                image = cv2.resize(image[int(y1):int(y2), int(x1):int(x2)], (224, 224))

                os.remove(os.path.join(root, f))
                cv2.imwrite(os.path.join(root, f), image)

                bar()

file_count = sum(len(files) for _, _, files in os.walk(file + "/gmm_left_hand_clusters"))

print("loading hand csv file...")
df = pd.read_csv(file + "/left_hand_keypoints.csv")
df = np.array(df)
df = df[:, 1:]

paths = pd.read_csv(file + "/left_hand_paths.csv")
paths = np.array(paths)
paths = paths[:, 1:]
print("hand csv file loaded!\n")

frame_no = 0
with alive_bar(file_count) as bar:
    for root, dirs, files in os.walk(file + "/gmm_left_hand_clusters"):
        for f in files:
            if f.endswith(".jpg"):
                image = cv2.imread(os.path.join(root, f))
                keypoints = df[int(f.split('.')[0])]

                x1 = np.min(keypoints[0::3]) * image.shape[1]
                x2 = np.max(keypoints[0::3]) * image.shape[1]
                y1 = np.min(keypoints[1::3]) * image.shape[0]
                y2 = np.max(keypoints[1::3]) * image.shape[0]

                if x2 - x1 >= y2 - y1:
                    d = (x2 - x1) * 0.1
                else:
                    d = (y2 - y1) * 0.1

                x1 -= d
                y1 -= d
                x2 += d
                y2 += d

                if x2 - x1 >= y2 - y1:
                    y1 -= ((x2 - x1) - (y2 - y1))/2
                    y2 += ((x2 - x1) - (y2 - y1))/2
                else:
                    x1 -= ((y2 - y1) - (x2 - x1))/2
                    x2 += ((y2 - y1) - (x2 - x1))/2
                
                if x1 < 0:
                    x1 = 0
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                if y1 < 0:
                    y1 = 0
                if y2 > image.shape[0]:
                    y2 = image.shape[0]

                image = cv2.resize(image[int(y1):int(y2), int(x1):int(x2)], (224, 224))

                os.remove(os.path.join(root, f))
                cv2.imwrite(os.path.join(root, f), image)

                bar()
