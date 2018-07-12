import csv
import cv2
import numpy as np

lines = []

#with open("C:\\Users\\hang\\Desktop\\Project_03_Behavioral_Cloning\\data\\driving_log.csv") as csvfile:
with open("./data_collected/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:

	# the path stored in the local machine
	source_path = line[0]

	# extract the filename
	filename = source_path.split('/')[-1]

	#current_path = 'C:\\Users\\hang\\Desktop\\Project_03_Behavioral_Cloning\\data\\IMG\\' + filename
	current_path = './data_collected/IMG/' + filename

	# load the image in BGR format using OpenCV
	image = cv2.imread(current_path)
	images.append(image)

	# load the steering measurement
	measurement = float(line[3])
	measurements.append(measurement)

# data augmentation
augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):

	augmented_images.append(image)
	augmented_measurements.append(measurement)

	# flip image horizontally
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)

	"""
	image_flipped = np.fliplr(image)
	measurement_flipped = -measurement
	"""


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense
# from keras.layers import Lambda     # Keras v2
from keras.layers.core import Lambda  # Keras v1.2.1
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D


# construct a regression network
model = Sequential()

## LeNet
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=3)

model.save('model.h5')



