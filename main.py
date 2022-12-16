"""
Number Identification Neural Network
by Shash Comandur | 12/16/22

This code is intended to build a neural network that can identify a hand drawn digit 0—9.
I wrote this using @NeuralNine's tutorial.
All credit to him for the code, this is simply for the sake of practice and my understanding of the topic!

This the code that evalutes the performances of the model, and runs it on my own samples.
The code that trains the model can be found in 'train.py'.
"""

# imports 
import cv2                  # for loading and processing images
import numpy as np          # for numpy arrays
import matplotlib.pyplot as plt    # for visualization of digits
import tensorflow as tf     # for neural network structure
import os

# from train.py, just so that this script has access to the test data
# ---------------------------------------------------------------------------------------------------
# load dataset
mnist = tf.keras.datasets.mnist

# split into training data & testing data 
# training data trains the model, testing data assesses model performance (typically an 80/20 split)
# x_train is the pixel data of an image, y_train is the number classification
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the pixel data to the range [0, 1] — current range is [0, 255], because images are b&w
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# ---------------------------------------------------------------------------------------------------

# train the model by running train.py
os.chdir('neural-network-number-identification')
os.system('python train.py')

# load the model
model = tf.keras.models.load_model('handwritten.model')

# evaluate the model — these metrics (loss & accuracy) tell us how good our model is
# we want a low loss and high accuracy — this model delivers loss and accuracy of around 0.07-0.1 and 0.97—0.98 respectively
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# now let's try to run the model on my own hand drawn sample digits, contained in the folder sample-digits
image_number = 1
while os.path.isfile(f"sample-digits/digit{image_number}.png"):
    try:
        # set up the image, convert to an array and invert it to white on black
        img = cv2.imread(f"sample-digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))

        # predict
        prediction = model.predict(img)
        print(f"This digit is most likely a {np.argmax(prediction)}.")

        # draw the image and display it
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1