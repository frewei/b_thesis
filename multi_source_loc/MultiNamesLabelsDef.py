azimuth_values = [270, 280, 290, 300, 310, 320, 330, 340, 350, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
elevation_values= [60, 45, 30, 20, 10, 0, -10, -20, -30, -45]
positions_dict = []
for az in azimuth_values:
  for el in elevation_values:
    positions_dict.append({'azimuth': az, 'elevation': el})
# print(positions_dict)
# print(len(positions_dict))
# print(positions_dict[0]['azimuth'])

import numpy as np
import os #through files
import re #regex


#code for loading all labels and file names for the generator
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
        pattern = r'Az_([-\d]{3})_El_([-\d]{3})'
        matches = re.findall(pattern, file_name)

        # Finds the indeces of the label's true classes
        file_index = [positions_dict.index({'azimuth': int(az), 'elevation': int(el)}) 
                        for az, el in matches]
        
        labels.append(file_index)

    names = np.array(file_names)
    labels = np.array(labels)
    np.savez(f'/home/frewei/multi_source_loc/name_labels_{step}.npz', names, labels)#end of function
    print("length of set", len(names))



# TRAIN FOLDER

# multi_folder = ['Cochleagrams_Two_SoundScenes_Train_FrederiqueWeiss_New', 
#                 'Cochleagrams_Two_SoundScenes_Train_FrederiqueWeiss_New2'] 

# labels_names_to_list_multi(multi_folder, positions_dict, 'train_multi')


# TEST FOLDER

test_folder = ['Cochleagrams_Two_SoundScenes_Test_FrederiqueWeiss_New'] 

labels_names_to_list_multi(test_folder, positions_dict, 'test_multi')
