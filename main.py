import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist #uses mnist data set

(x_train, y_train), (x_test, y_test) = mnist.load_data() #xdata is image and corresponding y coord is actual number

x_train = tf.keras.utils.normalize(x_train, axis = 1) #scales everything down to 0-1
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential() #uses basic neural network

#Input Layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) #Changes 2D array to 1D

#2 Hidden Layers
model.add(tf.keras.layers.Dense(128, activation='relu')) #Uses RelU activation function
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax')) #Output layer
#softmax makes sure every neuron added together = 1, whichever one is highest value is the
#predicted outcome

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics =['accuracy'])

model.fit(x_train, y_train, epochs=12) #Trains data using mnist data set, fits squiggly line to data
#epochs are how many times network sees the data


model.save('digitrecognition.model') #Saves model for reuse


#COMMENT OUT ABOVE CODE TO AVOID TRAINING SAME DATA EVERY TIME

model = tf.keras.models.load_model('digitrecognition.model')


image_num = 1
while os.path.isfile(f"digits/{image_num}.png"):
    try:
        img = cv2.imread(f"digits/{image_num}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        predict = model.predict(img) #Gives activation value
        print(f"Prediction: {np.argmax(predict)}") #Finds highest activation value in output layer
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error. Try setting the image resolutions to 28x28")
    finally:
        image_num += 1
