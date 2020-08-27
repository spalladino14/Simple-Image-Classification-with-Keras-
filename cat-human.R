library(reticulate)
library(tidyverse)
library(tensorflow)
library(keras)
library(magick)
use_condaenv("r-reticulate", required = TRUE)
################################
# Simple image classification using the Keras API
# This is just a way that felt natural to me but there are 
#other ways using Generator APIs in Keras
################################################################
#If we want to check out the cats and human pictures 
cat = image_read("C:\\Users\\spenc\\Desktop\\nn_pics2\\training\\01.jpg")
human = image_read("C:\\Users\\spenc\\Desktop\\nn_pics2\\training\\09.jpg")
plot(cat)
plot(human)

# Import Training and Test data
#Training
trainpics = list.files("C:\\Users\\spenc\\Desktop\\nn_pics2\\training", 
                       full.names = TRUE) 

#instead of using loops, the apply functions work great here 
loadpics = function(filenames) {
  a = lapply(filenames, image_load, grayscale = TRUE) #grayscale the image
  b = lapply(a, image_to_array) #turns it into an array
  c = lapply(b,image_array_resize, height = 100, width = 100) #resize
  d = normalize(c, axis = 1) #normalize to make small numbers 
  return(d)}

trainx =loadpics(trainpics) #loads training files

#Testing
testpics = list.files("C:\\Users\\spenc\\Desktop\\nn_pics2\\test", 
                      full.names = TRUE) 

testx = loadpics(testpics) # load test files

#make the labels for the training data
#0 is feral beast and 1 is human
trainy = c(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1)
testy  = c(0,0,1,1)
trainlabel = to_categorical(trainy)
testlabel = to_categorical(testy)

#Define
model1 = keras_model_sequential()
model1 %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 128, activation = 'relu')%>%
  layer_dense(units = 2, activation = 'softmax')

# Compile
model1 %>%
  compile (optimizer = 'adam',loss = 'categorical_crossentropy', 
           metrics = c('accuracy'))

# Fit
fit1 = model1 %>%
  fit(x = trainx, y = trainlabel, epochs = 20, batch_size=32 ,
      validation_split = .2, callbacks = callback_tensorboard("logs/run_a")) 
#the callback just lets me view the model in tensorboard
plot(fit1)
# Evaluate Training Data
model1 %>%
  evaluate(trainx,trainlabel)
#Evaluate Test Data
model1 %>%
  evaluate(testx,testlabel)
#Predict
predictedclasses1 = model1 %>%
  predict_classes(testx)
table(Prediction = predictedclasses1, Actual = testy)
#So we got 3 out of 4 of our test data guesses correctly. I think I 
#could have been more accurate with a bigger data set of photos. 

#tensorboard("logs/run_a")

save_model_tf(model1, "model1") #save the model
