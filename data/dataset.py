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

        self._length = 0
        self.to_pad = int(receptive_field / 2)
        self.train = train

        self.datas = []
        self.data_lens=[]
        for i in range(8000):
            file_name = str(i + 1).zfill(6)
            d_path = os.path.join(data_folder, file_name+'_data.npy')
            l_path = os.path.join(data_folder, file_name+'_label.npy')

            data = np.load(d_path)
            label = np.load(l_path)
            self._length += len(label)
            self.data_lens.append(len(label))

            p_data = np.pad(data, ((0, 0), (self.to_pad, self.to_pad)), 'constant', constant_values=0)

            self.datas.append((p_data, label))

            #test
            break

    def __getitem__(self, idx):

        current_files=None
        current_file_idx = 0
        total_len = 0
        for fid, _len in enumerate(self.data_lens):
            current_file_idx = idx - total_len
            total_len+=_len
            if idx<total_len:
                current_files = self.datas[fid]
                break

        p_data, label = current_files
        sample = p_data[:, current_file_idx:current_file_idx+self._receptive_field]
        sample = torch.Tensor(sample)

        label = label[current_file_idx]

        return sample, label

    def __len__(self):
        return self._length
