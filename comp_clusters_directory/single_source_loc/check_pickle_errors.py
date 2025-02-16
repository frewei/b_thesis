# Check if there are any corrupted files in the single-source folders 
import os
import pickle
import numpy as np

test = np.load('/home/frewei/single_source_loc/corrupt_name_labels_test_single.npz')['arr_0']
train = np.load('/home/frewei/single_source_loc/corrupt_name_labels_train_single.npz')['arr_0']

# TESTSET
print("checking testset..")
corrupted_files_test = []

for ID in test:
    try:
        np.load('/home/kikhei/' + ID, allow_pickle=True)
    except pickle.UnpicklingError:
        corrupted_files_test.append(ID)
    except Exception as e:
        print(f"Error processing file {ID}: {e}")
        corrupted_files_test.append(ID)
if not corrupted_files_test:
    print("No corrupted files found.")
else:
    print("amount of corrupts test", len(corrupted_files_test))
    corrupted_files_test = np.array(corrupted_files_test)
    np.savez('/home/frewei/single_source_loc/corrupted_tests.npz', corrupted_files_test)

#TRAINSET
print("checking trainset..")
corrupted_files_train = []

for ID in train:
    try:
        np.load('/home/kikhei/' + ID, allow_pickle=True)
    except pickle.UnpicklingError:
        corrupted_files_train.append(ID)
    except Exception as e:
        print(f"Error processing file {ID}: {e}")
        corrupted_files_train.append(ID)
if not corrupted_files_train:
    print("No corrupted files found.")
else:
    print("amount of corrupts train", len(corrupted_files_train))
    corrupted_files_train = np.array(corrupted_files_train)
    np.savez('/home/frewei/single_source_loc/corrupted_trains.npz', corrupted_files_train)

