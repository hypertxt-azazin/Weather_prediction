import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the updated CSV file
data = pd.read_csv(r"C:\Users\rma\OneDrive\Documents\GitHub\Weather_prediction\weather_data.csv")

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Normalize WindSpeed and Pressure for scaling purposes
scaler = MinMaxScaler()
data[['WindSpeed', 'Pressure']] = scaler.fit_transform(data[['WindSpeed', 'Pressure']])

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(data.index, data['Temperature'], 'bo-', label='Temperature')
ax1.plot(data.index, data['Humidity'], 'gx-', label='Humidity')
ax2.plot(data.index, data['WindSpeed'], 'y^-', label='WindSpeed (normalized)')
ax2.plot(data.index, data['Pressure'], 'rs-.', label='Pressure (normalized)')

ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (°C) / Humidity (%)', color='blue')
ax2.set_ylabel('WindSpeed / Pressure (normalized)', color='red')
plt.title("Weather Data: Temperature, Humidity, WindSpeed, and Pressure over Time")

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()
