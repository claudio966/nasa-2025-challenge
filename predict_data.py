import glob
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import xarray
import numpy as np


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
H, W, time_steps = no2_measured[0].shape[0], no2_measured[0].shape[1], 3

with tf.device('/CPU:0'):
    model = tf.keras.models.Sequential([
        tf.keras.layers.ConvLSTM2D(
            filters=8, kernel_size=(3,3),
            activation='relu',
            input_shape=(time_steps, H, W, 1),
            return_sequences=False,   # output only last frame
            padding='same'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='linear', padding='same')
    ])

x_input = no2_measured[0:15]
y_input = no2_measured[16]
x_input = x_input[np.newaxis, :, :, :, np.newaxis]
y_input = y_input[np.newaxis, :, :, np.newaxis]
model.compile(optimizer='adam', loss='mse')
model.fit(x_input, y_input)
predicted_matrix = model.predict(x_input)
predicted_matrix = np.squeeze(predicted_matrix)
y_input = np.squeeze(y_input)
predicted_matrix[mask_nan[16]==0] = np.nan # convert again to nan representation
y_input[mask_nan[16]==0] = np.nan

plt.imshow(predicted_matrix, cmap="viridis", origin="lower")
plt.colorbar(label="Value")
plt.title("Predicted values")
plt.xlabel("X index")
plt.ylabel("Y index")
plt.savefig("no2_predicted.pdf")
plt.close()
plt.imshow(y_input, cmap="viridis", origin="lower")
plt.colorbar(label="Value")
plt.title("Original values")
plt.ylabel("Y index")
plt.xlabel("X index")
plt.savefig("no2_original.pdf")

