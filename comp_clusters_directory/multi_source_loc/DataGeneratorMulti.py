# Code below is from url https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# about how to make your own keras data generator

# multi
import numpy as np
import keras
from sklearn.preprocessing import MultiLabelBinarizer

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=64, dim=(39,8000), n_channels=2,
                 n_classes=190, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 2), dtype=int)
        class_ind = np.array([i for i in range(190)]) # array of class indices
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('/home/kikhei/' + ID)['arr_0'].reshape(39, 8000, 2) #resahped naar channels last
            # Store class
            y[i] = self.labels[i]
        # one_hot_encode labels
        mlb = MultiLabelBinarizer(classes = class_ind)
        one_hot_y = mlb.fit_transform(y)
        return X, one_hot_y

