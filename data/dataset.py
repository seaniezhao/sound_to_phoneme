import os
import os.path
import math
import torch
import torch.utils.data
import numpy as np



class STPDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_folder,
                 receptive_field,
                 train=True):

        #           |----receptive_field----|
        # example:  | | | | | | | | | | | | |
        # target:               |
        self.data_folder = data_folder
        self._receptive_field = receptive_field

        self.data = np.load(self.data_folder+'/data.npy')
        self.label = np.transpose(np.load(self.data_folder+'/label.npy'))

        self._length = self.data.shape[1]
        self.to_pad = int(receptive_field/2)
        self.data = np.pad(self.data, ((0, 0), (self.to_pad, self.to_pad)), 'constant', constant_values=0)
        self.train = train


    def __getitem__(self, idx):

        sample = self.data[:, idx:idx+self._receptive_field]
        sample = torch.Tensor(sample)

        label, = np.where(self.label[:, idx] == 1)
        label = torch.LongTensor(label).squeeze()

        return sample, label

    def __len__(self):
        return self._length
