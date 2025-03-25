import pandas as mp 
import numpy as np
import tenserflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('weather_data.csv', parse_dates=['Date'])

# Select features and target variable
features = ['Humidity', 'Windspeed', 'Pressure']
target = ['Temperature']

# Normalize data
scaler = MinMaxScaler()
data[featues] = scaler.fit_transform(data[features])

# Prepare training and testing data
x= data[features].values
y= data[features].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

# Define the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(1)
])

#Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

