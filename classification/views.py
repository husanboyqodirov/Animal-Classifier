from django.shortcuts import render, redirect, get_object_or_404

import os
import numpy as np
import tensorflow as tf

#Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from classify.models import Classify

# Initializing the CNN
model = Sequential()

# Step 1 - Convolution
model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size= (2, 2)))

model.add(Convolution2D(32, 3, 3, activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full Connection
model.add(Dense(128, activation = 'relu'))

# Output layer
model.add(Dense(10, activation = 'softmax'))

#Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 : Fitting the CNN to the images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('media/training',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 class_mode='categorical')
validation_set = validation_datagen.flow_from_directory('media/validation',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            class_mode='categorical')

def home(request):
    return render(request, 'home.html')

def train(request):
    
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    model.load_weights(checkpoint_path)

    # model_fit = model.fit(training_set,
    #                  steps_per_epoch=3,
    #                  epochs=3000,
    #                  validation_data=validation_set, callbacks=[cp_callback])

    training = {'complete': 'Training successfully finished!'}

    return render(request, 'home.html', training)

def classify(request):

    Classify_obj = Classify.objects
    classify1 = Classify()
    classify1.image = request.FILES['filePath']
    classify1.save()

    dir_path ='media/test'

    img = image.load_img(dir_path+'/' + request.FILES['filePath'].name, target_size=(64,64))
    
    x = image.img_to_array(img)

    x = np.expand_dims(x,axis=0)

    val = model.predict(x)

    if val[0][0] == 1:
        animal = {
            'name': 'you are a butterfly !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }

    elif val[0][1] == 1:
        animal = {
            'name': 'you are a cat !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }

    elif val[0][2] == 1:
        animal = {
            'name': 'you are a chicken !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }

    elif val[0][3] == 1:
        animal = {
            'name': 'you are a giraffe !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }

    elif val[0][4] == 1:
        animal = {
            'name': 'you are a penguin !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }
    elif val[0][5] == 1:
        animal = {
            'name': 'you are a snake !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }
    elif val[0][6] == 1:
        animal = {
            'name': 'you are a spider !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }
    elif val[0][7] == 1:
        animal = {
            'name': 'you are a tiger !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }
    elif val[0][8] == 1:
        animal = {
            'name': 'you are a wolf !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }
    elif val[0][9] == 1:
        animal = {
            'name': 'you are a zebra !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }
    else:
        animal = {
            'name': 'you are a unknown !',
            'image': dir_path+'/' + request.FILES['filePath'].name
            }

    return render(request, 'home.html', animal)