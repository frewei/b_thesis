import numpy as np
import os

#SINGLE SOURCE CORRUPTED FILE DELETION
print("removing corrupted files from single source directory...")
corr_test = np.load('/home/frewei/single_source_loc/corrupted_tests.npz')['arr_0']
corr_train = np.load('/home/frewei/single_source_loc/corrupted_trains.npz')['arr_0']

dir_path = '/home/kikhei/Cochleagrams_Single_SoundScenes_Train_FrederiqueWeiss_New/'

print("deleting corrupted files from", dir_path)
print("path len pre: ", len(os.listdir(dir_path)))

for file in os.listdir(dir_path):
    path = dir_path + file
    fol_file = 'Cochleagrams_Single_SoundScenes_Train_FrederiqueWeiss_New/' + file
    if fol_file in corr_test:
        os.remove(path)
    if fol_file in corr_train:
        os.remove(path)
        
print("path len post: ", len(os.listdir(dir_path)))


dir_path = '/home/kikhei/Cochleagrams_Single_SoundScenes_Train_FrederiqueWeiss_New2/'
print("deleting corrupted files from", dir_path)
print("path len pre: ", len(os.listdir(dir_path)))

for file in os.listdir(dir_path):
    path = dir_path + file
    fol_file = 'Cochleagrams_Single_SoundScenes_Train_FrederiqueWeiss_New2/' + file
    if fol_file in corr_test:
        os.remove(path)
    if fol_file in corr_train:
        os.remove(path)

print("path len post: ", len(os.listdir(dir_path)))


print("corrupted files are deleted from single source directory")





#MULTI SOURCE CORRUPTED FILE DELETION
print("removing corrupted files from multi source directory...")

corr_train = np.load('/home/frewei/multi_source_loc/corrupted_multi.npz')['arr_0']

dir_path = '/home/kikhei/Cochleagrams_Two_SoundScenes_Train_FrederiqueWeiss_New/'
print("deleting corrupted files from", dir_path)

print("path len pre: ", len(os.listdir(dir_path)))

for file in os.listdir(dir_path):
    path = dir_path + file
    fol_file = 'Cochleagrams_Two_SoundScenes_Train_FrederiqueWeiss_New/' + file
    if fol_file in corr_train:
        os.remove(path)

print("path len post: ", len(os.listdir(dir_path)))

dir_path = '/home/kikhei/Cochleagrams_Two_SoundScenes_Train_FrederiqueWeiss_New2/'
print("deleting corrupted files from", dir_path)

print("path len pre: ", len(os.listdir(dir_path)))

for file in os.listdir(dir_path):
    path = dir_path + file
    fol_file = 'Cochleagrams_Two_SoundScenes_Train_FrederiqueWeiss_New2/' + file
    if fol_file in corr_train:
        os.remove(path)

print("path len post: ", len(os.listdir(dir_path)))

print("corrupted files are deleted from multi source directory")
