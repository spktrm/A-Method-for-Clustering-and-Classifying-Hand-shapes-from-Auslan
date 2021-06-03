import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd

from glob import glob
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

X = pd.read_csv("handshapes/hand_keypoints.csv")
y = np.array(X)[:, -1]
X = np.array(X)[:, 1:-1]

model_path = "hand_models/modelv4.9"
model = keras.models.load_model(model_path)

if os.path.exists(model_path + "/state.txt"):
    state = int(np.loadtxt(model_path + "/state.txt"))
else:
    state = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=state)

class_dir = "handshapes/*/"
class_names = names = [os.path.basename(x[0:-1]) for x in glob(class_dir)]

model.summary()

# tf.keras.utils.plot_model(model, to_file="/" + model_path + "/model.png")

y_pred = np.argmax(model.predict(X_test), axis=1)

print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names, annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')

# logs = pd.read_csv(model_path + "/logs.csv")
# logs = np.array(logs)[:, 1]

# epochs_range = range(int(logs[0]))

# acc = logs[1].split(" ")
# for i in range(acc.count("")):
#     acc.remove("")
# for i in range(len(acc)):
#     acc[i] = acc[i].replace("[", "")
#     acc[i] = acc[i].replace("]", "")
#     acc[i] = float(acc[i])

# val_acc = logs[2].split(" ")
# for i in range(val_acc.count("")):
#     val_acc.remove("")
# for i in range(len(val_acc)):
#     val_acc[i] = val_acc[i].replace("[", "")
#     val_acc[i] = val_acc[i].replace("]", "")
#     val_acc[i] = float(val_acc[i])

# loss = logs[3].split(" ")
# for i in range(loss.count("")):
#     loss.remove("")
# for i in range(len(loss)):
#     loss[i] = loss[i].replace("[", "")
#     loss[i] = loss[i].replace("]", "")
#     loss[i] = float(loss[i])

# val_loss = logs[4].split(" ")
# for i in range(val_loss.count("")):
#     val_loss.remove("")
# for i in range(len(val_loss)):
#     val_loss[i] = val_loss[i].replace("[", "")
#     val_loss[i] = val_loss[i].replace("]", "")
#     val_loss[i] = float(val_loss[i])

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
plt.tight_layout()
plt.show()

