import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Input, Dense, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(len(x_train), 28,28,1)/255
x_test = x_test.reshape(len(x_test), 28,28,1)/255

noise_factor = 0.5

x_train_noisy = np.clip(x_train + noise_factor*np.random.normal(size=x_train.shape), 0., 1.)
x_test_noisy = np.clip(x_test + noise_factor*np.random.normal(size=x_test.shape), 0., 1.)

input_image = Input(shape=(28,28,1))
x = Flatten()(input_image)
x = Dense(128, activation='relu')(x)
encoded = Dense(64, activation='relu')(x)
encoded = Dense(64, activation='relu')(encoded)
x = Dense(128,activation='relu')(encoded)
x = Dense(784, activation='sigmoid')(x)
output_image = Reshape((28,28,1))(x)

autoencoder = Model(input_image, output_image)
autoencoder.compile(optimizer = 'adam',loss = 'binary_crossentropy')
autoencoder.summary()
autoencoder.fit(x_train_noisy, x_train, epochs=20, batch_size=128, shuffle=True, validation_data=(x_test_noisy, x_test))

predictions = autoencoder.predict(x_test_noisy)

plt.figure(figsize=(20,6))
for i in range(10):
    plt.subplot(3,10,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.title('Noisy images')

    plt.subplot(3,10,i+10+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.title('Original images')

    plt.subplot(3,10,i+20+1)
    plt.imshow(predictions[i].reshape(28,28))
    plt.title('Denoised images')

plt.tight_layout()
plt.show()
