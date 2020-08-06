'Developed by Amirhossein Rezaei'
#Maximum validation accuracy: 98.97% (on 15 epochs; withoun any image processing)
#On this dataset, the ratio between training and test data is 2:1.
#This result was obtaind with 50,000 training images and 25,000 testing images. 
#V 1.0.0 , hyperparameter tuning is needed.


from keras import backend as K
import os
import keras
from PIL import Image
from captcha.image import ImageCaptcha  # pip install captcha
import matplotlib.pyplot as plt
import random
import os
from scipy import misc
import numpy as np
from matplotlib.pyplot import imread
from tkinter.filedialog import askdirectory  
from keras.utils import to_categorical
from keras.models import Sequential , model_from_json
from keras.layers import Dense, Conv2D , Flatten , MaxPooling2D , Dropout, LSTM , ConvLSTM2D , BatchNormalization ,Conv3D
from keras import Model


number = ['0','1','2','3','4','5','6','7','8','9']

MAX_CAPTCHA = 1

WIDTH=28
HEIGHT=28

Number_of_train=50000

def get_char_set():
	return number

def get_char_set_len():
	return len(get_char_set())

def get_captcha_size():
	return MAX_CAPTCHA

def get_y_len():
	return MAX_CAPTCHA*get_char_set_len()

def get_width():
    return WIDTH

def get_height():
    return HEIGHT



def random_captcha_text(char_set=get_char_set(), captcha_size=get_captcha_size()):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text
x_train=[]
li=[]
def gen_captcha_text_and_image(i):
  global y_train
  image = ImageCaptcha(width=WIDTH, height=HEIGHT, font_sizes=[25])

  captcha_text = random_captcha_text()
  captcha_text = ''.join(captcha_text)
  #print(captcha_text)
  li.append(int(captcha_text))
  y_train=to_categorical(li)
  captcha = image.generate(captcha_text)
  captcha_image = Image.open(captcha)
  captcha_image = np.array(captcha_image)
  x_train.append(captcha_image)
  return captcha_text, captcha_image
if __name__ == '__main__':

        NUMBER_OF_CAPTCHAS_GENERATED=int(Number_of_train*1.5)
        for i in range(NUMBER_OF_CAPTCHAS_GENERATED):     
                text, image = gen_captcha_text_and_image(i)



#shows an example of the generated captcha:
#print(np.shape(x_train[0]),y_train[0])
print(y_train[0])
plt.imshow(x_train[0], interpolation='nearest')
plt.show()



x_t= np.array(x_train)
x_t=np.reshape(x_t,[NUMBER_OF_CAPTCHAS_GENERATED,1, HEIGHT, WIDTH, 3])
y_t=np.array(y_train)
classes= np.shape(y_train)[1]



#create model

model = Sequential()

model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                   input_shape=(1,HEIGHT, WIDTH, 3),
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
 

model.fit(x_t[:Number_of_train], y_t[:Number_of_train],validation_data=(x_t[Number_of_train:], y_t[Number_of_train:]) ,batch_size=128 ,epochs=15)


X_te=x_t[Number_of_train:]

import math
#n is Index of image in testing dataset.
n=24

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

