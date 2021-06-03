import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random
import matplotlib.pyplot as plt
import pandas as pd

from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import regularizers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class_dir = "handshapes/*/"
class_names = names = [os.path.basename(x[0:-1]) for x in glob(class_dir)]

X = pd.read_csv("handshapes/hand_keypoints.csv")

y = np.array(X)[:, -1]
X = np.array(X)[:, 1:-1]

state = random.randint(0, 2**32-1)
epochs = 1000
model_path = "hand_models/modelv5.1"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=state)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=state)

# Model Initialization
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 3,)),
    tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(41, activation='softmax')
])

model.summary()

# Compiling the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model Fitting
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=32,
    validation_data=(X_val, y_val),
    use_multiprocessing=True
)

model.save(model_path)
np.savetxt(model_path + "/state.txt", np.array([str(state)]), fmt="%s")

acc = np.array(history.history['accuracy'])
val_acc = np.array(history.history['val_accuracy'])
loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

logs = pd.DataFrame(np.array([epochs, acc, val_acc, loss, val_loss]))
logs.to_csv(model_path + "/logs.csv")

