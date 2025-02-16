#Generate predictions of model
import os #through files
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # GPU 1 on computational clusters

import numpy as np
import keras
from DataGeneratorClass import DataGenerator

# Array of dictionaries of the labels at their indeces
azimuth_values = [270, 280, 290, 300, 310, 320, 330, 340, 350, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
elevation_values= [60, 45, 30, 20, 10, 0, -10, -20, -30, -45]
positions_dict = []
for az in azimuth_values: 
  for el in elevation_values:
    positions_dict.append({'azimuth': az, 'elevation': el})

# Load test data
X_test = np.load('/home/frewei/single_source_loc/name_labels_final_test_single.npz')['arr_0']
y_test = np.load('/home/frewei/single_source_loc/name_labels_final_test_single.npz')['arr_1']

params_test = {'dim': (39,8000),
          'batch_size': 8,
          'n_classes': len(positions_dict),
          'n_channels': 2,
          'shuffle': False}

#DataGenerator
testing_generator = DataGenerator(X_test, y_test, **params_test)


# Predicting the model with the best weights (checkpoint models)
print("pred good model...")
model = keras.saving.load_model('/home/frewei/home/frewei/single_source_loc/tmp/ckpt/checkpoint6_bs8.model.keras')

predictions = model.predict(testing_generator)
np.savez('/home/frewei/final_predictions/goodEpoch_SingleSource.npz', predictions)


# Predicting the overfitted model
print("pred bad model...")

bad_model = keras.saving.load_model('/home/frewei/single_source_loc/models/model_NewSS6_DiffArch_LR_0.000001_epoch_16_BS_8.keras')

predictions = bad_model.predict(testing_generator)
np.savez('/home/frewei/final_predictions/badEpoch_SingleSource.npz', predictions)