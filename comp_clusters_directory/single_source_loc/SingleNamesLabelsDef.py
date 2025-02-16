# Run code for creating snap-shot of the files in the current single-source folders

import numpy as np
import os #through files
import re #regex
from sklearn.model_selection import train_test_split

# Array of dictionaries of the labels at their indeces
azimuth_values = [270, 280, 290, 300, 310, 320, 330, 340, 350, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
elevation_values= [60, 45, 30, 20, 10, 0, -10, -20, -30, -45]
positions_dict = []
for az in azimuth_values:
  for el in elevation_values:
    positions_dict.append({'azimuth': az, 'elevation': el})


def labels_names_to_list_multi(folders, positions_dict, step):
    file_names = []
    labels = []

    for folder in folders:
      filepath = '/home/kikhei/' + folder
      for file in  os.listdir(filepath):
        batch_file = folder + '/' + file
        file_names.append(batch_file)
      #print("total files for ",folder ," ...with length ", len(file_names))

    for file_name in file_names:
        #regex to find the azimuth and elevation values
        az_match = re.search(r'Az_([-\d]{3})', file_name)
        el_match = re.search(r'El_([-\d]{3})', file_name)
        
        az_value = int(az_match.group(1)) if az_match else None
        el_value = int(el_match.group(1)) if el_match else None

        file_index = positions_dict.index({'azimuth': az_value, 'elevation': el_value})
        labels.append(file_index)

    #create npz files for the names_labels
    names = np.array(file_names)
    labels = np.array(labels)
    np.savez(f'/home/frewei/single_source_loc/name_labels_{step}.npz', names, labels)#end of function



# # TRAIN FOLDER

# single_folder = ['Cochleagrams_Single_SoundScenes_Train_FrederiqueWeiss_New',
#                  'Cochleagrams_Single_SoundScenes_Train_FrederiqueWeiss_New2'] 

# labels_names_to_list_multi(single_folder, positions_dict, 'final_full_single')




# # TEST FOLDER

test_folder = ['Cochleagrams_Single_SoundScenes_Test_FrederiqueWeiss_New']

labels_names_to_list_multi(test_folder, positions_dict, 'final_test_single')



# TRAIN TEST SPLIT OF TRAIN FOLDER DURING PILOT TESTS

# X_train_val_test = np.load('/home/frewei/single_source_loc/name_labels_full_single.npz')['arr_0']
# y_train_val_test = np.load('/home/frewei/single_source_loc/name_labels_full_single.npz')['arr_1']

# print("length of set", len(X_train_val_test))

# X_train_val, X_test, y_train_val, y_test = train_test_split(X_train_val_test, y_train_val_test, 
#                                                             train_size = 0.95, 
#                                                             stratify = y_train_val_test)
# np.savez('/home/frewei/single_source_loc/name_labels_test_single.npz', X_test, y_test)
# np.savez('/home/frewei/single_source_loc/name_labels_train_single.npz', X_train_val, y_train_val)
