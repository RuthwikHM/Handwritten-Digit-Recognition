#Importing libraries
import numpy as np
import keras
#Importing dataset,required layers and models
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Activation
from keras.layers import Conv2D,MaxPooling2D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing import image
print('Loading data\n')
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_test1=x_test
print('Training sample')
plt.imshow(x_train[0],cmap='gray')
plt.show()
print('Test sample')
plt.imshow(x_test[0],cmap='gray')
plt.show()
#Reshaping the arrays
x_train=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_test=x_test.reshape(x_test.shape[0],28,28,1).astype('float32')
#Normalizing the input
x_train/=255
x_train/=255
#Converting class vectors to binary class matrices
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=y_test.shape[1]
#Creating the model
model=Sequential()
model.add(Conv2D(32,(4,4),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))
#Compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Fitting data to the model')
#Fitting data to model
model.fit(x_train,y_train,batch_size=200,epochs=10,validation_split=0.3)
print('Evaluating the test data on model')
score=model.evaluate(x_test,y_test,batch_size=512)
print('Test accuracy=',score[1])
pre=model.predict(x_test)
n=int(input('Enter a number between 0 and 9999'))
prediction=np.argmax(pre[n])
print('The number is ',prediction)
plt.imshow(x_test1[n],cmap='gray')
plt.show()
