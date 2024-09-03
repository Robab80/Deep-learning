from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
#for noise creation
import numpy as np
(X_train, _), (X_test,_)=mnist.load_data()

X_train=X_train.astype('float32')/255
X_test=X_test.astype('float32')/255
#noise
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
#normalization to be in 0-1
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)


print(X_train.shape)#show dimensions but input should be a vector so we reshape
print(X_test.shape)
X_train=X_train.reshape(len(X_train),784)
X_test=X_test.reshape(len(X_test),784) # put pixels in a vector with size 28*28=784

X_train_noisy = X_train_noisy.reshape(len(X_train_noisy), 784)
X_test_noisy = X_test_noisy.reshape(len(X_test_noisy), 784)

print(X_train.shape)
print(X_test.shape)
input_image=Input(shape=(784,))
encoder=Dense(units=32,activation='relu')(input_image)
decoder=Dense(units=784, activation='sigmoid')(encoder)
autoencoder=Model(input_image,decoder)
autoencoder.summary()
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(X_train_noisy,X_train, epochs=30,batch_size=256)
#batch_size: the number of images simultanously enter the model
encoder_model=Model(input_image, encoder)
encoder_model.summary()
pred=autoencoder.predict(X_test_noisy)
encoded_images=encoder_model.predict(X_test_noisy)
# show 10 samples of images
plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded image
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_images[i].reshape(8,4))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # display reconstruction
    ax = plt.subplot(3, 20, 2*20 +i+ 1)
    plt.imshow(pred[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
