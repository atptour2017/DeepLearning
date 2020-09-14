import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
x=train_images.reshape(60000, 28 * 28)

print(train_images.shape)
y=x.astype('float32')/255

from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
print(train_images.shape)
train_images = train_images.reshape((60000, 28 * 28))
print(train_images.shape)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
from tensorflow.keras.utils import to_categorical
print(train_labels)
train_labels = to_categorical(train_labels)
print(train_labels)
print(train_labels.shape)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

