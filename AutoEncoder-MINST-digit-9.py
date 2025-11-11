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
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(len(x_train), -1)/255
x_test = x_test.reshape(len(x_test), -1)/255

input_image = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_image)
encoded = Dense(128, activation='relu', activity_regularizer=regularizers.l1(2e-4))(encoded)
encoded = Dense(64, activation='relu')(encoded)
latent = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(latent)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
output_image = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_image, output_image)

optimizer = AdamW(learning_rate=0.0001, weight_decay=5e-4)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(x_train, x_train, epochs=20, validation_data=(x_test, x_test))

predictions = autoencoder.predict(x_test[:10])

plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.title('Original')

    plt.subplot(2,10,i+1+10)
    plt.imshow(predictions[i].reshape(28,28))
    plt.title('Reconstructed')

plt.show()
