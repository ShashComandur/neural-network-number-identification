"""
Number Identification Neural Network
by Shash Comandur | 12/16/22

This code is intended to build a neural network that can identify a hand drawn digit 0—9.
I wrote this using @NeuralNine's tutorial.
All credit to him for the code, this is simply for the sake of practice and my understanding of the topic!

This is the code that trains the number idenfication model.
"""

# imports 
import tensorflow as tf     # for neural network structure

# load dataset
mnist = tf.keras.datasets.mnist

# split into training data & testing data 
# training data trains the model, testing data assesses model performance (typically an 80/20 split)
# x_train is the pixel data of an image, y_train is the number classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the pixel data to the range [0, 1] — current range is [0, 255], because images are b&w
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# create the model
model = tf.keras.models.Sequential()

# add layers, flatten 28x28 image into a 784 length column vector
model.add(tf.keras.layers.Flatten(input_shape=(28 ,28)))

# add two base dense layer, with rectify linear unit (relu) activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# add final dense layer, with 10 units, because this is the output — 10 corresponding to digits to 0—9
# activation is softmax — this makes sure that all 10 neurons add up to one
# this can be interpreted as a confidence value — how likely a given image is a certain digit
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# compile the model!
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model! (train it)
# 3 epochs guarantees that the model will run through the training data 3 times
model.fit(x_train, y_train, epochs=3)
model.save('handwritten.model')