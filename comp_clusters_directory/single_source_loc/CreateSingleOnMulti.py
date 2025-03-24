import tensorflow
from tensorflow import keras
from keras.models import load_model
from keras import layers
from keras.layers import Conv2D, ReLU, BatchNormalization, MaxPooling2D, Dense, Dropout, Activation, Softmax, Flatten

#load single-source model
loaded_model = load_model('/home/frewei/home/frewei/single_source_loc/tmp/ckpt/checkpoint6_bs8.model.keras')


#load weights
weights = loaded_model.get_weights()


#architecture of single-source model, but with sigmoid activation
data_format = "channels_last" #2is before 39 and 8000

inputs = keras.Input(shape=(39, 8000, 2))

x = layers.Conv2D(filters = 16, kernel_size = (1, 2), data_format = data_format)(inputs)
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


#set weights from single-source model for architecture with sigmoid activation
model.set_weights(weights)

#save model in multi-source folder
model.save('/home/frewei/single_source_loc/models/single_on_multi_model.keras')#model has to be 