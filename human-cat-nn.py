#Spencer Palladino
#Classification using NN in Python

#Import packages I may use
import tensorflow.keras as keras
from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

# Import Training and Test data
#Training
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
trainpath = glob.glob("C:\\Users\\spenc\\Desktop\\nn_pics2\\training\\*.jpg")
trainpics = []
for img in trainpath:
    n = load_img(img, grayscale = True)
    trainpics.append(n)
#display an image
trainpics[0].show() 

#Lets make a function to turn the pictures into a usable array
def loadpics(file, shape):
	# load the image with size
	image = load_img(file, target_size = shape, grayscale=True)
	# convert to array
	image = img_to_array(image)
	# normalize pixels
	image = image.astype('float32')
	image /= 255.0
	return image

trainarray = []
for img in trainpath:
    n = loadpics(img, (100,100))
    trainarray.append(n)
#To view the array
print(trainarray[0])
X_train = np.array(trainarray,dtype = np.float32)

#Testing
testpath = glob.glob("C:\\Users\\spenc\\Desktop\\nn_pics2\\test\\*.jpg")
testarray = [] 
for img in testpath:
    n = loadpics(img, (100,100))
    testarray.append(n)
X_test = np.array(testarray,dtype = np.float32)
#make the labels for the training data

#0 is feral beast and 1 is human
trainy = (0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1)
testy  = (0,0,1,1)
trainy = np_utils.to_categorical(trainy)
testy = np_utils.to_categorical(testy)
#Define
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
# Compile
model.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ('accuracy'))

# Fit
model.fit(X_train, trainy, epochs=25)  # train the model
# Evaluate Training Data

val_loss, val_acc = model.evaluate(X_train, trainy)
#Predict
val_loss, val_acc = model.evaluate(X_test, testy)



