# Check if there are any corrupted files in the single-source folders 
import os
import pickle
import numpy as np

train = np.load('/home/frewei/multi_source_loc/corrupt_name_labels_train_multi.npz')['arr_0']

#only checking trainset as some files corrupted before the testset started generating
print("checking trainset..")
corrupted_files_train = []

for ID in train:
    try:
        np.load('/home/kikhei/' + ID, allow_pickle=True)
    except pickle.UnpicklingError:
        corrupted_files_train.append(ID)
    except Exception as e:
        # Handle other exceptions (optional)
        print(f"Error processing file {ID}: {e}")
        corrupted_files_train.append(ID)
if not corrupted_files_train:
    print("No corrupted files found.")
else:
    print("amount of corrupts train", len(corrupted_files_train))
    corrupted_files_train = np.array(corrupted_files_train)
    np.savez('/home/frewei/multi_source_loc/corrupted_multi.npz', corrupted_files_train)

