import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
         if (path_data == 'DHF1K_dataset'):
             path_to_file = 'DHF1K_num_frame_train.csv'#'Atari_num_frame_train.csv', 'r'))]
         else:
             path_to_file = 'Atari_num_frame_train.csv'
         csv_reader = csv.reader(open(path_to_file, 'r'))
         list_of_tuples = list(map(tuple, csv_reader))  #list of (#samples, file_name)
         num_frame = []
         for (n_samples, name) in list_of_tuples:
             num_frame.append((int(n_samples), name))
         self.list_num_frame = num_frame

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        file_name = self.list_num_frame[idx][1]
        #file_name = '%04d'%(idx+1)
        path_clip = os.path.join(self.path_data, 'video', file_name)
        path_annt = os.path.join(self.path_data, 'annotation', file_name, 'maps')

        start_idx = np.random.randint(1, self.list_num_frame[idx][0]-self.len_snippet+1) #(0, ..) to keep 1st frame

        v = np.random.random()
        clip = []
        for i in range(self.len_snippet):
            img = cv2.imread(os.path.join(path_clip, '%06d.png'%(start_idx+i+1)))
            img = cv2.resize(img, (384, 224))
            img = img[...,::-1]
            if v < 0.5:
                img = img[:, ::-1, ...]
            clip.append(img)

        annt = cv2.imread(os.path.join(path_annt, '%06d.png'%(start_idx+self.len_snippet)), 0)
        annt = cv2.resize(annt, (384, 224))
        if v < 0.5:
            annt = annt[:, ::-1]

        return transform(clip), torch.from_numpy(annt.copy()).contiguous().float(), (file_name, '%06d.png'%(start_idx+self.len_snippet))

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