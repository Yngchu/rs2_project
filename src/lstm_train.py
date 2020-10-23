import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

data_dir = "csv_file/"
stock_name = "trajectory2"
epoch = 400
stock_dataset = pd.read_csv(data_dir + stock_name + ".csv", sep=',')
stock_dataset.head()
dataset_x = stock_dataset.get('x')
dataset_y = stock_dataset.get('y')
dataset = []
for i in range(0, 440):
    if 89 < i <100 or 189 < i <200 or 289 < i <300 or 389 < i <400:
        continue
    x_data = [dataset_x[i]/100, dataset_y[i]/100]
    dataset.append(x_data)
dataset = np.array(dataset, dtype=float)
dataset = dataset.reshape(40, 10, 2)

label = []
for i in range(5, 445):
    if 94 < i <105 or 194 < i <205 or 294 < i <305 or 394 < i <405:
        continue
    y_data = [dataset_x[i]/100, dataset_y[i]/100]
    label.append(y_data)
label = np.array(label, dtype=float)
label = label.reshape(40, 20)

# split data and labels into train and test
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=4)
print y_test.shape
model =Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape = (x_train.shape[1],2)))
model.add(GRU(units=50))
model.add(Dense(20))
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))

results = model.predict(x_test)
results = results.reshape(results.shape[0],10,2)
y_test = y_test.reshape(8, 10, 2)
results = results.reshape(8, 10, 2)
a_axis=[]
b_axis=[]
a_r_axis = []
b_r_axis = []
for i in range(8):
    for j in range(10):
        a_x = y_test[i][j][0]
        a_axis.append(a_x)
        b_x = y_test[i][j][1]
        b_axis.append(b_x)
        a_r = results[i][j][0]
        a_r_axis.append(a_r)
        b_r = results[i][j][1]
        b_r_axis.append(b_r)

compare = plt.figure(1)
plt.scatter(a_axis,b_axis,c='g')
plt.scatter(a_r_axis,b_r_axis,c='r')

loss = plt.figure(2)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

model.save('10step_lstm_model.h5')