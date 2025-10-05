import glob
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import xarray
import numpy as np
from scipy import constants
import copy

# threshold for no2 concentration in ppb
ppb_no2 = [[0, 53], [54, 100], [101, 360],
                         [361, 649], [650, 1249], [1250, 2049], [2050, 3049]]
# corresponding AQI
aqi_no2 = [[0, 50], [51, 100], [101, 150],
                         [151, 200], [201, 300], [301, 400], [401, 500]]

ppm_o3 = [[0, 0.124], [0.125, 0.164], [0.165, 0.204], [0.205, 0.404], [0.405, 0.604]]
aqi_03 = [[0, 50], [101, 150], [151, 200], [201, 300], [301, 500]]

use_o3 = True
use_no2 = False

if use_o3:
    pp = ppm_o3
    aqi = aqi_03
elif use_no2:
    pp = ppb_no2
    aqi = aqi_no2

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable all GPUs
no2_measured = []
counter = 0
for filename in np.sort(glob.glob("data/*.nc"))[:20]: # get only 20 samples
    counter += 1
    ds = xarray.open_dataset(filename)
    no2_measured.append(ds["weight"].values)
    print("Sample: ", counter)

no2_measured = np.array(no2_measured, dtype=np.float16)
print(no2_measured.shape, np.float16)
mask_nan = (~np.isnan(no2_measured)).astype(np.float32)
no2_measured = np.nan_to_num(no2_measured, nan=0.0)

# Example: HxW=32x32, time_steps=4
H, W, time_steps = no2_measured[0].shape[0], no2_measured[0].shape[1], 5

with tf.device('/CPU:0'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.ConvLSTM2D(
            filters=32, kernel_size=(3,3),
            activation='relu',
            input_shape=(time_steps, H, W, 1),
            return_sequences=False,   # output only last frame
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='relu', padding='same')
    ])

counter = 0
predicted_matrices = []
for i in range(time_steps, len(no2_measured)):
    if i == len(no2_measured) - 1:
        break
    x_input = no2_measured[0+counter:time_steps+counter]
    counter += 1
    y_input = no2_measured[time_steps+counter]
    x_input = x_input[np.newaxis, :, :, :, np.newaxis]
    y_input = y_input[np.newaxis, :, :, np.newaxis]
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_input, y_input)
    predicted_matrix = model.predict(x_input)
    predicted_matrices.append(np.squeeze(predicted_matrix))
    
predicted_matrices = np.array(predicted_matrices)
surface_number_density = (predicted_matrices * 10e15) / 1e5
air_number_density = 101325 / (constants.k * 288)
# evaluate the concentration in ppb
mixing_ratio = (surface_number_density / air_number_density) * 1e9 

for i, _ in enumerate(pp):
    mask = (pp[i][0] <= mixing_ratio) & (mixing_ratio <= pp[i][1])

    mixing_ratio[mask] = (
        (aqi[i][1] - aqi[i][0]) / (pp[i][1] - pp[i][0])
    ) * mixing_ratio[mask] + aqi[i][0]
    aqi_measured = mixing_ratio

print(aqi_measured.shape)

aqi_measured[0][mask_nan[0]==0] = np.nan # convert again to nan representation

plt.imshow(aqi_measured[0], cmap="viridis", origin="lower")
plt.colorbar(label="Value")
plt.title("Predicted values")
plt.xlabel("X index")
plt.ylabel("Y index")
plt.savefig("no2_predicted_aqi.pdf")
plt.close()

