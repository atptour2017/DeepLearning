import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import models

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
x=train_images.reshape(60000, 28 * 28)
print(x.shape)
print(x[0])
print(train_images.shape)
y=x.astype('float32')/255
print(y[0])