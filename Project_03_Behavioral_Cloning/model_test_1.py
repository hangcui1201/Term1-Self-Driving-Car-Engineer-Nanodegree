
import csv
import cv2
import numpy as np

lines = []

#with open("../data/driving_log.csv") as csvfile:
with open("C:\\Users\\hang\\Desktop\\Project_03_Behavioral_Cloning\\data\\driving_log.csv") as csvfile:
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
	current_path = 'C:\\Users\\hang\\Desktop\\Project_03_Behavioral_Cloning\\data\\IMG\\' + filename
	#current_path = '../data/IMG/' + filename

	# load the image in BGR format using OpenCV
	image = cv2.imread(current_path)
	images.append(image)

	# load the steering measurement
	measurement = float(line[3])
	measurements.append(measurement)

# extract the center images and corresponding steering angles
X_train = np.array(images)
y_train = np.array(measurements)

#print(images[0])
#print(len(measurements))

from keras.models import Sequential
from keras.layers import Flatten, Dense

# construct a regression network
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')

exit()


