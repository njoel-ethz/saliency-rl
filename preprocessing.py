import os
import csv
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def atari_reader(path_indata, file_name): #called if some flag (flagfile) is false
    path_zipped = 'Atari_zipped_version'

    #unzip

    path_video = os.path.join(path_indata, 'video', file_name)
    path_annt = os.path.join(path_indata, 'annotation', file_name, 'maps')

    #copy images to new location and rename them to counter
    #create saliency maps from .txt file
        f = open(....txt, 'r')
        full_data = f.readlines()
        f.close()
        #if null: interpolate from previous and next (if exists)
        #gaussian blurr
        #write img to annt file
    #set flag to true
        return 0

# from gist.github.com/keithweaver/70df4922fec74ea87405b83840b45d57
def split_video_to_frames(old_path, file_name):
    FPS = 25

    # Playing video from file: 384, 224
    cap = cv2.VideoCapture(os.path.join(old_path, file_name + '.mp4'))
    cap.set(cv2.CAP_PROP_FPS, FPS)

    new_path = os.path.join(old_path, file_name)
    try:
        if not os.path.exists(new_path):
            os.makedirs(new_path)
    except OSError:
        print('Error: Creating directory of data')

    currentFrame = 0
    while (True):
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success: break
        # Saves image of the current frame in jpg file
        file_name = '%04d.png' % (currentFrame + 1)
        name = os.path.join(new_path, file_name)
        # print ('Creating... ' + name)
        frame = cv2.resize(frame, (384, 224))
        cv2.imwrite(name, frame)
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return new_path

def main():
    ''' preprocessing of input data '''
    #split into frames if needed
    #read in atari data and save accordingly

if __name__ == '__main__':
    main()
