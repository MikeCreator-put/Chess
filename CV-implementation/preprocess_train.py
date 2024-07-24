import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_data(data_dir, img_size=(48, 48)):
    images = []
    labels = []
    label_map = {folder: idx for idx, folder in enumerate(os.listdir(data_dir))}
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label_map[folder])
    
    images = np.array(images).astype('float32') / 255.0
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels)
    
    return images, labels, label_map

data_dir = r'D:\personal projects\Chess\CV-implementation\collected data'
images, labels, label_map = load_data(data_dir)




def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (48, 48, 1)
num_classes = len(label_map)
model = create_model(input_shape, num_classes)

model.fit(images, labels, epochs=10, validation_split=0.2)
model.save('expression_model.h5')
