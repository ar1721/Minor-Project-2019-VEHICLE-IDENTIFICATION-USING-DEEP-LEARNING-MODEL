"""
Created on Wed Apr 17 05:40:21 2019

AMAN RAJ
169103011
Minor Project
"""

print ('AMAN RAJ')
print('169103011') 


from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense






train_dir = 'DATASET/train'

test_dir ='DATASET/test'

trsam = 8000

tesam = 2000

epochs =10

batch_size = 32



if K.image_data_format() == 'channels_first':

    input_shape = (3, 128, 128)

else:

    input_shape = (128, 128, 3)



model = Sequential()

#input layer
model.add(Conv2D(32, (3, 3), input_shape=input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

#Output Layer

model.add(Dense(6))

model.add(Activation('softmax'))


"""
adam or rmsprop
"""


# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



# this is the augmentation configuration we will use for testing

test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(128, 128),

    batch_size=batch_size,

    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=(128, 128),

    batch_size=batch_size,

    class_mode='categorical')

model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model.fit_generator(

    train_generator,

    steps_per_epoch=trsam // batch_size,

    epochs=epochs,

   validation_data=test_generator,

    validation_steps=tesam // batch_size)






model.summary()
model.get_weights()
model.layers

model.save('vechile.h5')
with open('vechile.json', 'w') as f:
    f.write(model.to_json())








from keras.models import model_from_json

# Model reconstruction from JSON file
with open('vechile128.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights('vechile128.h5')

from skimage.io import imread
from skimage.transform import resize
import numpy as np
 
class_labels = {v: k for k, v in train_generator.class_indices.items()}



i = 1
while i < 45:
    j=str(i)
    img = imread('Prediction/'+j+'.jpg') 
    img = resize(img,(128,128)) 
    img = np.expand_dims(img,axis=0) 
    prediction = model.predict_classes(img);
    if prediction == 0:
        print ("Background")
    elif prediction == 1:
        print ("Bus")
    elif prediction == 2:
        print ("Car")
    elif prediction == 3:
        print ("Non Motorized Vechile")
    elif prediction == 4:
        print ("Truck")
    elif prediction == 5:
        print ("Two-WHeeler")
    i=i+1;