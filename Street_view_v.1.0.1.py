'Developed by Amirhossein Rezaei'
#Maximum validation accuracy: 94.65% (on 15 epochs; without any image processing)
#V 1.0.1 , hyperparameter tuning is needed.

from google.colab import drive
import numpy as np
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
%matplotlib inline

from keras import backend as K
import os
import keras
from PIL import Image
import os
from scipy import misc
from matplotlib.pyplot import imread
from keras.utils import to_categorical
from keras.models import Sequential , model_from_json
from keras.layers import Dense, Conv2D , Flatten , MaxPooling2D , Dropout, LSTM , ConvLSTM2D , BatchNormalization ,Conv3D


train_raw = loadmat('/content/drive/My Drive/train_32x32.mat')
test_raw = loadmat('/content/drive/My Drive/test_32x32.mat')


train_images = np.array(train_raw['X'])
test_images = np.array(test_raw['X'])

train_labels = train_raw['y']
test_labels = test_raw['y']

train_images = np.moveaxis(train_images, -1, 0)
test_images = np.moveaxis(test_images, -1, 0)


train_images = train_images.astype('float64')
test_images = test_images.astype('float64')

train_images /= 255.0
test_images /= 255.0

train_images = np.reshape(train_images,[73257,1,32, 32, 3])
test_images =  np.reshape(test_images,[26032,1,32, 32, 3])


train_labels = train_labels.astype('int64')
test_labels = test_labels.astype('int64')

lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)

 
model = Sequential()
 
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                   input_shape=(1,32,32, 3),
                   padding='same', return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
 
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())
 
 
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
 
 
model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=False))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Conv2D(128, kernel_size=(5, 5), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.4))
model.add(Flatten())
 
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
 
 
model.fit(train_images,train_labels , validation_data=(test_images, test_labels) ,batch_size=128 ,epochs=15)



import math
#n is Index of image in testing dataset.
n=2
X_te=test_images


#First convlstm layer
#Enter number of filters:
nf1=64
inp= model.inputs
print("\n \n First convlstm layer")
f1=Model(inputs= inp, outputs= model.layers[0].output).predict(X_te[n:n+1])

fig1=plt.figure(figsize=(9, 9))

for i in range(nf1):
  s=f1[0,0,:,:,i]
  fig1.add_subplot(math.ceil(nf1/8),8,i+1)
  plt.imshow(s, interpolation='nearest') 
plt.show()



#Second convlstm layer
#Enter number of filters:
nf2=32
inp2= model.inputs
print("\n \n Second convlstm layer")
f2=Model(inputs= inp, outputs= model.layers[3].output).predict(X_te[n:n+1])

fig2=plt.figure(figsize=(9, 9))

for i in range(nf2):
  s=f2[0,0,:,:,i]
  fig2.add_subplot(math.ceil(nf1/8),8,i+1)
  plt.imshow(s, interpolation='nearest') 
plt.show()



#Third convlstm layer
#Enter number of filters:
nf3=32
inp3= model.inputs
print("\n \n Third convlstm layer")
f3=Model(inputs= inp, outputs= model.layers[5].output).predict(X_te[n:n+1])

fig3=plt.figure(figsize=(9, 9))

for i in range(nf3):
  s=f3[0,0,:,:,i]
  fig3.add_subplot(math.ceil(nf1/8),8,i+1)
  plt.imshow(s, interpolation='nearest') 
plt.show()





#Convolutional layer
#Enter number of filters:
nf4=128

print("\n \n Convolutional layer")
f4=Model(inputs= inp, outputs= model.layers[11].output).predict(X_te[n:n+1])

fig4=plt.figure(figsize=(9, 9))

for i in range(nf4):
  s=f4[0,:,:,i]
  fig4.add_subplot(math.ceil(nf4/8),8,i+1)
  plt.imshow(s, interpolation='nearest') 
plt.show()
