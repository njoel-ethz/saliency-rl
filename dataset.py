import os
import csv
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    snippet = snippet.view(-1,3,snippet.size(1),snippet.size(2)).permute(1,0,2,3)
    return snippet

class DHF1KDataset(Dataset):
    def __init__(self, path_data, len_snippet):
         self.path_data = path_data
         self.len_snippet = len_snippet
         self.list_num_frame = [int(row[0]) for row in csv.reader(open('Atari_num_frame_train.csv', 'r'))]

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        file_name = '%04d'%(idx+1)
        #path_clip = os.path.join(self.path_data, 'video', file_name)

        ###
        temp = os.path.join(self.path_data, 'video', file_name)
        if os.path.isdir(temp):
            path_clip = temp
            print(file_name)
        else:
            path_clip = null #path_clip = split_video_to_frames(os.path.join(self.path_data, 'video'), file_name)
            print(file_name + ' missing, preprocessing.py needed')

        #path_clip = atari_reader(os.path.join(self.path_data, 'video'), file_name)
        ###

        path_annt = os.path.join(self.path_data, 'annotation', file_name, 'maps')

        start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)

        v = np.random.random()
        clip = []
        for i in range(self.len_snippet):
            img = cv2.imread(os.path.join(path_clip, '%06d.png'%(start_idx+i+1)))
            if not os.path.isfile(os.path.join(path_clip, '%06d.png'%(start_idx+i+1))):
                print(('%06d.png'%(start_idx+i+1)) + ' missing')
            img = cv2.resize(img, (384, 224))
            img = img[...,::-1]
            if v < 0.5:
                img = img[:, ::-1, ...]
            clip.append(img)
        
        annt = cv2.imread(os.path.join(path_annt, '%06d.png'%(start_idx+self.len_snippet)), 0)
        annt = cv2.resize(annt, (384, 224))
        if v < 0.5:
            annt = annt[:, ::-1]

        return transform(clip), torch.from_numpy(annt.copy()).contiguous().float()

# from gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch
