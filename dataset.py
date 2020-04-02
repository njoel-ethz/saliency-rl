import os
import csv
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    snippet = snippet.view(-1,3,snippet.size(1),snippet.size(2)).permute(1,0,2,3)
    return snippet

#from gist.github.com/keithweaver/70df4922fec74ea87405b83840b45d57
def precompute_video(old_path, file_name):
    FPS = 25

	# Playing video from file:
    cap = cv2.VideoCapture(os.path.join(old_path, file_name + '.mp4'))
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    new_path = os.path.join(old_path, file_name)
    try:
        if not os.path.exists(new_path):
            os.makedirs(new_path)
    except OSError:
        print ('Error: Creating directory of data')

    currentFrame = 0
    while(True):
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success: break
        # Saves image of the current frame in jpg file
        file_name = '%04d.png'%(currentFrame+1)
        name = os.path.join(new_path, file_name)
        print ('Creating... ' + name)
        cv2.imwrite(name, frame)
        currentFrame += 1

        
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return new_path

class DHF1KDataset(Dataset):
    def __init__(self, path_data, len_snippet):
         self.path_data = path_data
         self.len_snippet = len_snippet
         self.list_num_frame = [int(row[0]) for row in csv.reader(open('DHF1K_num_frame_train.csv', 'r'))]

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        file_name = '%04d'%(idx+1)
        #path_clip = os.path.join(self.path_data, 'video', file_name)

        ###
        path_clip = precompute_video(os.path.join(self.path_data, 'video'), file_name)
        ###

        path_annt = os.path.join(self.path_data, 'annotation', file_name, 'maps')

        start_idx = np.random.randint(0, self.list_num_frame[idx]-self.len_snippet+1)

        v = np.random.random()
        clip = []
        for i in range(self.len_snippet):
            img = cv2.imread(os.path.join(path_clip, '%04d.png'%(start_idx+i+1)))
            img = cv2.resize(img, (384, 224))
            img = img[...,::-1]
            if v < 0.5:
                img = img[:, ::-1, ...]
            clip.append(img)

        annt = cv2.imread(os.path.join(path_annt, '%04d.png'%(start_idx+self.len_snippet)), 0)
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