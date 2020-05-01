import os
import csv
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tarfile
import zipfile
import shutil

def main():
    ''' preprocessing of input data '''
    # split into frames if needed
    # read in atari data and save accordingly

    if not cv2.imwrite(os.path.join('Atari_dataset','annotation','test.png'), np.zeros((210, 160, 3), np.uint8)):
        print('failed')


if __name__ == '__main__':
    main()
