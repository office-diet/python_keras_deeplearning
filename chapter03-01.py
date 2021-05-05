# #################################################
# chapter 03-01
# #################################################

## example
from tensorflow.keras import models
from tensorflow.keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation="relu", input_shape=(784, )))
model.add(layers.Dense(10, activation="softmax"))

from tensorflow.keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss="mse", metrics=["accuracy"])
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

model.fit(train_images, train_labels, batch_size=128, epochs=5)


input_tensor = layers.Input(shape=(784, ))
x = layers.Dense(32, activation="relu")(input_tensor)
output_tensor = layers.Dense(10, activation="softmax")(x)
model = models.Model(inputs=input_tensor, outputs=output_tensor)
