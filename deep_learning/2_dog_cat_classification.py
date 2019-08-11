import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow._api.v1.keras.preprocessing.image import ImageDataGenerator
from tensorflow._api.v1.keras.datasets import cifar10
from tensorflow._api.v1.keras.callbacks import TensorBoard
import time

'''
    数据集地址
        https://www.microsoft.com/en-us/download/details.aspx?id=54765
'''

DATADIR = "./Datasets/PetImages"
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE =50
X_PICKLE_FILE = 'X.pickle'
Y_PICKLE_FILE = 'y.pickle'
NAME= 'Cats-vs-Dog-cnn-64x2-{}'.format(int(time.time()))

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) 
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                #plt.imshow(img_array, cmap='gray')
                #plt.show()
                #print(img_array)
            except Exception as e:
                pass
    print(len(training_data))
    # 要随机一下
    random.shuffle(training_data)

    X = []
    y = []

    for feathers, lable in training_data:
        X.append(feathers)
        y.append(lable)
    
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    with open(X_PICKLE_FILE, "wb") as pickle_out:
        pickle.dump(X, pickle_out)
    
    with open(Y_PICKLE_FILE, "wb") as pickle_out:
        pickle.dump(y, pickle_out)

    return (X, y)


def load_training_data():
    if os.path.exists(X_PICKLE_FILE) and os.path.exists(Y_PICKLE_FILE):
        X = pickle.load(open(X_PICKLE_FILE, 'rb'))
        y = pickle.load(open(Y_PICKLE_FILE, 'rb'))
        return (X, y)
    else:
        return None

data = load_training_data()
if not data:
    data = create_training_data()

X, y = data
X = X / 255.0
print(len(X))

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))


model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=1, validation_split=0.3, callbacks=[tensorboard])
