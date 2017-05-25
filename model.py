import os
import csv
import pandas as pd
import numpy as np
from scipy import ndimage
from scipy.misc import imresize
import json
from skimage.exposure import adjust_gamma

# Load training files provided by Udacity
samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line.extend(['data'])
        samples.append(line)

del(samples[0])


recovery_samples = []
with open('./recovery_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line.extend(['recovery_data'])
        recovery_samples.append(line)

# Load recovery files
del(recovery_samples[0])
samples.extend(recovery_samples)

# Drop samples with small steering values
filtered_samples = []
for sample in samples:
    cut_value = 0.35
    del_rate = 0.3
    if abs(float(sample[3])) < cut_value:
        if np.random.random() < del_rate:
            continue
    filtered_samples.append(sample)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(filtered_samples, test_size=0.2, random_state=42)


import cv2
import numpy as np
import sklearn

def process_image(image):
    # Cut out top and bottom of an image to remove parts that are not essential
    image = image[70:-25, :, :]
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    return image

# Generator function:
# input are either traning or testing
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            # images and angles will renew every yield cycle
            images = []
            angles = []

            for batch_sample in batch_samples:

                source_path = batch_sample[0]
                dir_name = batch_sample[-1]
                filename = source_path.split('/')[-1]
                filename = filename.strip('center')

                # Depending on where the image is located, dir_name will locate them.
                center_path = './' + dir_name + '/IMG/' + 'center' + filename
                left_path = './' + dir_name + '/IMG/' + 'left' + filename
                right_path = './' + dir_name + '/IMG/' + 'right' + filename

                image_center = cv2.imread(center_path)
                image_center = process_image(image_center)


                image_left = cv2.imread(left_path)
                image_left = process_image(image_left)


                image_right = cv2.imread(right_path)
                image_right = process_image(image_right)


                correction = 0.25
                steering_center = float(batch_sample[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                image_flipped = np.fliplr(image_center)
                steering_flipped = -steering_center

                images.extend([image_center, image_left, image_right, image_flipped])

                angles.extend([steering_center, steering_left, steering_right, steering_flipped])

            X_train = np.array(images)
            X_train = adjust_gamma(X_train)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D, SpatialDropout2D, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

model = Sequential()
model.add(BatchNormalization(axis=1, input_shape=(64,64,3)))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2), activation='elu'))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(SpatialDropout2D(0.2))
model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
model.add(SpatialDropout2D(0.2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))
model.add(Dropout(0.5))
model.add(Dense(1))

# Compile model with adam optimizer and learning rate of .0001
adam = Adam(lr=0.0001)
model.compile(loss='mse',  optimizer=adam)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=6, verbose=1)
model.save('model.h5')
print('Model Saved!')
