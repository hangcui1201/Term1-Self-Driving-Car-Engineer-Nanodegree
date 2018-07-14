import csv
import cv2
import os
import numpy as np

car_images = []
steering_angles = []

samples = []
with open("./data_collected/driving_log.csv", 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

import sklearn
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.25 # this is a parameter to tune, 21
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:

                center_angle = float(batch_sample[3])
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                path = './data_collected/IMG/'

                center_path = path + batch_sample[0].split('/')[-1]
                left_path = path + batch_sample[1].split('/')[-1]
                right_path = path + batch_sample[2].split('/')[-1]

                # read image in the BGR format
                center_image = cv2.imread(center_path)
                left_image = cv2.imread(left_path)
                right_image = cv2.imread(right_path)

                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                
                images.append(center_image)
                angles.append(center_angle)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)

                images.append(left_image)
                angles.append(left_angle)
                images.append(cv2.flip(left_image,1))
                angles.append(left_angle*-1.0)

                images.append(right_image)
                angles.append(right_angle)
                images.append(cv2.flip(right_image,1))
                angles.append(right_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
# from keras.layers import Lambda     # Keras v2
from keras.layers.core import Lambda  # Keras v1.2.1
from keras.layers import Convolution2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
# from keras.layers import Cropping2D             # Keras v2
from keras.layers.convolutional import Cropping2D # Keras v1.2.1

model = Sequential()

## NVIDIA Net: input 160x320x3 -> 160x320x3
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# remove the top 70 pixels and the bottom 25 pixels
# 160x320x3 -> 65x320x3
model.add(Cropping2D(cropping=((70,25),(0,0))))

# channels(filters):24, kernel: 5x5, stride: (2,2), padding(default): valid
# 65x320x3 -> 31x158x24
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))

# channels:36, kernel: 5x5, stride: (2,2), padding(default): valid
# 31x158x24 -> 14x77x36
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))

# channels:48, kernel: 5x5, stride: (2,2), padding(default): valid
# 14x77x36  -> 5x37x48
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))

# channels:64, kernel: 3x3, stride: (2,2), padding(default): valid
# 5x37x48 -> 3x35x64
model.add(Convolution2D(64,3,3, activation="relu"))

# channels:64, kernel: 3x3, stride: (2,2), padding(default): valid
# 3x35x64 -> 1x33x64
model.add(Convolution2D(64,3,3, activation="relu"))

model.add(Flatten())
model.add(Dense(100, activation="relu")) # fully-connected layer: 100 neurons
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))  # fully-connected layer: 50 neurons
model.add(Dropout(0.5))
model.add(Dense(10, activation="relu"))  # fully-connected layer: 10 neurons
model.add(Dense(1))

model.compile(loss='mse', optimizer=Adam(lr=0.001))
model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), nb_epoch=15)

model.save('model.h5')



