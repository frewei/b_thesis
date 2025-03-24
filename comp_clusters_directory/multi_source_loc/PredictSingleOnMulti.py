import os #through files
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # GPU 1

import numpy as np
import keras
from DataGeneratorMulti import DataGenerator

azimuth_values = [270, 280, 290, 300, 310, 320, 330, 340, 350, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
elevation_values= [60, 45, 30, 20, 10, 0, -10, -20, -30, -45]
positions_dict = []
for az in azimuth_values: 
  for el in elevation_values:
    positions_dict.append({'azimuth': az, 'elevation': el})

X_test = np.load('/home/frewei/multi_source_loc/name_labels_test_multi.npz')['arr_0']
y_test = np.load('/home/frewei/multi_source_loc/name_labels_test_multi.npz')['arr_1']

params_test = {'dim': (39,8000),
          'batch_size': 8,    
          'n_classes': len(positions_dict),
          'n_channels': 2,
          'shuffle': False}

testing_generator = DataGenerator(X_test, y_test, **params_test)

print("start prediction")

other_model = keras.saving.load_model('/home/frewei/single_source_loc/models/single_on_multi_model.keras')

predictions = other_model.predict(testing_generator)
np.savez('/home/frewei/final_predictions/single_on_multi.npz', predictions)