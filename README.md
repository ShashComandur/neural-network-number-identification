# neural-network-number-identification
 A neural network to identify handwritten numbers.

### Author
Shash Comandur  
comandus@email.sc.edu  
shashcomandur.com

# About
This code is intended to build a neural network that can identify a hand drawn digit 0—9. I wrote this using @NeuralNine's tutorial.

All credit to him for the code, this is simply for the sake of practice and my understanding of the topic!

# Structure
The repository contains two scripts, train.py & evaluate.py.

train.py contains all of the code the starts and trains the neural network using the MNIST dataset of handwritten numbers. 

evaluate.py, when run, will execute train.py and calculate the loss and accuracy of the neural network. It will also run the samples, written by me, that are contained in the folder sample-digits through the neural network, and write its guesses to the terminal.