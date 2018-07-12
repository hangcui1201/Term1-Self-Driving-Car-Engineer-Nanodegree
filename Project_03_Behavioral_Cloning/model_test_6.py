import csv
import cv2
import numpy as np

car_images = []
steering_angles = []

with open("./data_collected/driving_log.csv", 'r') as csvfile:

    reader = csv.reader(csvfile)

    for row in reader:

    	# get steering angle
        steering_center = float(row[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.20 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        path = './data_collected/IMG/' # fill in the path to your training IMG directory

        center_path = path + row[0].split('/')[-1]
        left_path = path + row[1].split('/')[-1]
        right_path = path + row[2].split('/')[-1]

        # read image in the BGR format
        center_image = cv2.imread(center_path)
        left_image = cv2.imread(left_path)
        right_image = cv2.imread(right_path)

        # add images and angles to data set
        car_images.append(center_image)
        car_images.append(left_image)
        car_images.append(right_image)

        steering_angles.append(steering_center)
        steering_angles.append(steering_left)
        steering_angles.append(steering_right)


# data augmentation
augmented_images, augmented_measurements = [], []

for image, measurement in zip(car_images, steering_angles):

	augmented_images.append(image)
	augmented_measurements.append(measurement)

	# flip image horizontally
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
# from keras.layers import Lambda     # Keras v2
from keras.layers.core import Lambda  # Keras v1.2.1
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
# from keras.layers import Cropping2D             # Keras v2
from keras.layers.convolutional import Cropping2D # Keras v1.2.1

# construct a regression network
model = Sequential()

## NVIDIA Net: input 3@160x320
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# remove the top 70 pixels and the bottom 25 pixels
model.add(Cropping2D(cropping=((70,25),(0,0))))
# channels:24, kernel: 5x5
model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
# channels:36, kernel: 5x5
model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
# channels:48, kernel: 5x5
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
# channels:64, kernel: 3x3
model.add(Convolution2D(64,3,3, activation="relu"))
# channels:64, kernel: 3x3
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())  # flatten to 
model.add(Dense(100)) # fully-connected layer: 100 neurons
model.add(Dropout(0.5))
model.add(Dense(50))  # fully-connected layer: 50 neurons
model.add(Dropout(0.5))
model.add(Dense(10))  # fully-connected layer: 1 neurons
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

model.save('model.h5')



