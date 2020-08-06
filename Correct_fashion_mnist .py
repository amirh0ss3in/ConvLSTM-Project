'Developed by Amirhossein Rezaei'
#Maximum validation accuracy: 91.38% (on 15 epochs; withoun any image processing)
#V 1.0.0 , hyperparameter tuning is needed.


from keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential , model_from_json
from keras.layers import Dense, Conv2D , Flatten , MaxPooling2D , Dropout, LSTM , ConvLSTM2D , BatchNormalization ,Conv3D
from keras import Model

X_tr = X_train.reshape(-1,1,28,28,1)
X_te = X_test.reshape(-1,1,28,28,1)
#X_tr=X_tr/255.0
#X_te=X_te/255.0
 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
 
model = Sequential()
 
model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                   input_shape=(1,28,28,1),
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
 
 
model.fit(X_tr, y_train, validation_data=(X_te, y_test) ,batch_size=128 ,epochs=5)


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

