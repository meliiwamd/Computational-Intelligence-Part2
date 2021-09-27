#write your code here
# We Use Keras Library in This Question

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# =========== Keras Model =========== #

Model = Sequential()
Model.add(Dense(40, input_shape=(1,), activation = 'sigmoid'))
Model.add(Dense(20, input_shape=(1,), activation = 'sigmoid'))
Model.add(Dense(1))

# =========== Complie & Run =========== #

# Model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["acc"])
Model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])



# Run part

# =========== Data =========== #

x = np.arange(-300, 300).reshape(-1,1) / 100
y = np.sin(x)

# x_test, y_test = 
# validation_data = (x_test, y_test)

# =========== Summary =========== #

print(Model.summary())

for i in range(2):
    Model.fit(x, y, epochs = 1000, batch_size = 8)
Preds = Model.predict(x)

# =========== Accuracy & Error Ploting =========== #

plt.plot(x, Preds, 'g--')
plt.ylabel('MLP')
plt.xlabel('X Range')
plt.show()
 


