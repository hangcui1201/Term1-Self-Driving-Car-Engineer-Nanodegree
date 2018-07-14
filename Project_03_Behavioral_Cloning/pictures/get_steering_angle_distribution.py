import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt

csv_file = 'driving_log.csv'

df = pd.read_csv(csv_file)

steering_angles = df['steering'].values

steering_angles = np.round(steering_angles, decimals=4)

x_values = sorted(list(set(steering_angles)))
y_occurrence = dict(collections.Counter(steering_angles))

y_values = []
for i in range(len(x_values)):
	y_values.append(y_occurrence[x_values[i]])

print(x_values)
print(y_values)

plt.figure()
plt.scatter(x_values, y_values)
plt.plot(x_values, y_values)
plt.title('Steering Angle Distribution')
plt.show()
