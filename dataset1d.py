from __future__ import print_function

import math

import torch
from PIL import Image
import os
import os.path
import numpy as np
import sys
import torch.utils.data as data

import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE, ADASYN


class Dataset(data.Dataset):

    def __init__(self, root, train=True, num_splits=None,
                 transform=None, add_transform=None, target_transform=None,
                 download=False, ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.add_transform = add_transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.num_splits = num_splits

        # now load the picked numpy arrays
        if self.train:
            df = pd.read_csv(self.root, header=None)
            
            train_data = df.iloc[:, 0:121].to_numpy()
            train_data = preprocessing.scale(train_data)
            self.train_labels = df.iloc[:, 121].to_numpy(dtype=np.long)
            
            self.train_data = torch.FloatTensor(np.expand_dims(train_data,1))
        else:
            df = pd.read_csv(self.root, header=None)
            
            test_data = df.iloc[:, 0:121].to_numpy()
            test_data = np.log(test_data + 1)
            test_data = preprocessing.scale(test_data)
            print(len(test_data))

            self.test_data = torch.FloatTensor(np.expand_dims(test_data,1))
            print(self.test_data.shape)

            self.test_labels = df.iloc[:, 121].to_numpy(dtype=np.long)
            
        # else:
        #     df = pd.read_csv(self.root, header=None)

        #     test_data = df.iloc[:, 0:121].to_numpy()
        #     test_data = np.log(test_data + 1)

        #     df_k = pd.read_csv('KDDTest+.txt',header=None)
        #     print(len(df_k))
        #     df_k = df_k.to_numpy()
        #     df_index = df_k[:,-1]!=21

        #     test_data = preprocessing.scale(test_data)#[df_index]
        #     print(len(test_data))

        #     self.test_data = torch.FloatTensor(np.expand_dims(test_data,1))
        #     print(self.test_data.shape)
        #     self.test_labels = df.iloc[:, 121].to_numpy(dtype=np.long)#[df_index]
        #     print(self.test_labels.shape)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
        
        
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

