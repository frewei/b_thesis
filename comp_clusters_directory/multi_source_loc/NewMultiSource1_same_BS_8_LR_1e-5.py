#NewMS1_same_BS_8_LR_1e-5 

import os #through files
os.environ["CUDA_VISIBLE_DEVICES"] = "7" # GPU 2

os.environ['TF_DETERMINISTIC_OPS'] = '1' # set seed using the NVIDIA GPU documentation

# Imports
import numpy as np
import re #regex
from sklearn.model_selection import train_test_split #split data into test and train
import keras
import tensorflow as tf 

# Doublecheck if running on correct GPU
from tensorflow.python.client import device_lib##
print("GPU TENSORFLOW INFO: ", device_lib.list_local_devices())

import tensorboard
#%load_ext tensorboard
#import datetime
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, ReLU, BatchNormalization, MaxPooling2D, Dense, Dropout, Activation, Softmax, Flatten
from sklearn.model_selection import train_test_split
from DataGeneratorMulti import DataGenerator


azimuth_values = [270, 280, 290, 300, 310, 320, 330, 340, 350, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
elevation_values= [60, 45, 30, 20, 10, 0, -10, -20, -30, -45]
positions_dict = []
for az in azimuth_values: 
  for el in elevation_values:
    positions_dict.append({'azimuth': az, 'elevation': el})
      
X_train_val = np.load('/home/frewei/multi_source_loc/name_labels_train_multi.npz')['arr_0']
y_train_val = np.load('/home/frewei/multi_source_loc/name_labels_train_multi.npz')['arr_1']
# X_test = np.load('/home/frewei/multi_source_loc/name_labels_test_multi.npz')['arr_0']
# y_test = np.load('/home/frewei/multi_source_loc/name_labels_test_multi.npz')['arr_1']

first_label = []
for t in y_train_val:
    first_label.append(t[0])
first_label = np.array(first_label)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify = first_label)

                                                           
params_fit = {'dim': (39,8000),
          'batch_size': 8,
          'n_classes': len(positions_dict),
          'n_channels': 2,
          'shuffle': True}
# params_test = {'dim': (39,8000),
#           'batch_size': 16,
#           'n_classes': len(positions_dict),
#           'n_channels': 2,
#           'shuffle': False}


# Generators
training_generator = DataGenerator(X_train, y_train, **params_fit) 
validation_generator = DataGenerator(X_val, y_val, **params_fit)
# testing_generator = DataGenerator(X_test, y_test, **params_test)

data_format = "channels_last" #2is before 39 and 8000

inputs = keras.Input(shape=(39, 8000, 2))
x = layers.Conv2D(filters = 16, kernel_size = (1, 2), data_format = data_format, input_shape=(39,8000, 2))(inputs)
x = layers.MaxPooling2D(pool_size=(1, 2), padding = 'valid', data_format = data_format)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(filters = 32, kernel_size = (2, 4), data_format = data_format)(x)
x = layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid', data_format = data_format)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Conv2D(filters = 64, kernel_size = (3, 16), data_format = data_format)(x)
x = layers.MaxPooling2D(pool_size=(1, 2), padding = 'valid', data_format = data_format)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(rate = 0.5)(x)
outputs = layers.Dense(190, activation='sigmoid')(x)

model = keras.Model(inputs = inputs, outputs = outputs)

model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.00001), loss='binary_crossentropy', metrics=['accuracy'])

log_dir = "/home/frewei/logs/multi_source/" + "NewMS1_same_BS_8_LR_1e-5"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_filepath = "/home/frewei/multi_source_loc/tmp/ckpt/multi_checkpoint_bs8_lr1e-5.model.keras"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                               patience=3,
                                               mode="max",
                                               restore_best_weights=True, 
                                               start_from_epoch = 5)

print("training model...")
history = model.fit(x = training_generator,
                    epochs = 15,
                    validation_data=validation_generator,
                    callbacks=[tensorboard_callback, model_checkpoint_callback, early_stopping])


# print("predicting model...")
# predictions = model.predict(testing_generator)
# np.savez('/home/frewei/multi_source_loc/predictions/predictions_MS1_new.npz', predictions)


print("saving model...")
model.save('/home/frewei/multi_source_loc/models/model_NewMS1_same_BS_8_LR_1e-5.keras')

print("end of script, you did it:)")
