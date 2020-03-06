import os

import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class SwitchData(Dataset):
    def __init__(self, filename):
        """
        :param folder_dataset: str
        :param T: int
        :param symbols: list of str
        :param use_columns: bool
        :param start_date: str, date format YYY-MM-DD
        :param end_date: str, date format YYY-MM-DD
        """
        
        history_np=np.load(filename + '.npy',allow_pickle=True)
        self.history = history_np[()]

        self.scaler_train = MinMaxScaler()
        self.scaler_test = MinMaxScaler()
        self.nswitches = self.history['SWITCHES'].shape[1]


        self.switch = self.history['SWITCHES']
        self.z0 = np.expand_dims(self.history['Z'][:,0], axis=1)
        self.price = self.history['PRICES']
        self.numpy_data = np.hstack((self.switch, self.z0, self.price))
        #self.train_data = torch.FloatTensor(self.scaler.fit_transform(self.numpy_data))
        
        self.x_all = torch.FloatTensor(self.scaler_train.fit_transform(self.numpy_data[:,(self.nswitches):]))
        self.y_all = torch.FloatTensor(self.scaler_test.fit_transform(self.numpy_data[:,:(self.nswitches)]))


    def __getitem__(self, index):

        x = self.x_all[index,:]
        y = self.y_all[index,:]
        return x, y

    def __len__(self):
        return self.x_all.shape[0]