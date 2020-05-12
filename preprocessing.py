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


def atari_reader(path_indata):  # called if some flag (flagfile) is false
    counter = 0
    number_of_frames = []
    video = os.path.join(path_indata, "video")
    annotation = os.path.join(path_indata, 'annotation')
    num_frame_path = 'Atari_num_frame_train.csv'


    # unzip
    if not os.path.exists(annotation):
        os.makedirs(annotation)
    for file in os.listdir(path_indata):
        if file.endswith("zip"):
            current_path = os.path.join(path_indata, file)
            game_path = current_path[:-4]
            print(game_path)
            with zipfile.ZipFile(current_path, 'r') as zip_ref:
                zip_ref.extractall(path_indata)
            for game_file in os.listdir(game_path):
                # print("     "+game_file)
                if game_file.endswith("tar.bz2"):
                    sample_name = game_file.split('_')[1]
                    # untar
                    current_game_path = os.path.join(game_path, game_file)
                    tar = tarfile.open(current_game_path, "r:bz2")
                    tar.extractall(video)
                    tar.close()

                    # renaming videos
                    counter += 1
                    print('-- -- '+str(counter)+' -- --')
                    print(game_file.split('.')[0])
                    temp = os.path.join(video, game_file[:-8])
                    sample_path = os.path.join(video, '%04d' % counter)
                    if os.path.isdir(sample_path):
                        shutil.rmtree(sample_path)
                    os.rename(temp, sample_path)
                    path_annt = os.path.join(annotation, '%04d' % counter, 'maps')
                    if os.path.isdir(os.path.join(annotation,'%04d' % counter)):
                        shutil.rmtree(os.path.join(annotation,'%04d' % counter)) #ignore_errors=True
                    os.makedirs(os.path.join(annotation,'%04d' % counter))
                    os.makedirs(path_annt)

                    for frame in os.listdir(sample_path):
                        seq_number = frame.split('_')[2]
                        seq_number = int(seq_number[:-4])
                        os.rename(os.path.join(sample_path, frame), os.path.join(sample_path, '%06d' %seq_number + '.png'))
                        #print(frame + " ---> " + '%06d' %seq_number)
                        #print(cv2.imread(os.path.join(sample_path, '%06d' %seq_number + '.png')).shape) #(210, 160, 3)

                    #creating maps
                    path_txt = current_game_path[:-8] + '.txt'
                    f = open(path_txt, 'r')
                    lines_data = f.readlines()
                    f.close()
                    full_data = ''.join(lines_data)
                    full_data = full_data.replace('\n', '')
                    full_data = full_data.split(sample_name)
                    full_data.pop(0)  # frame_id,episode_id,score,duration(ms),unclipped_reward,action,gaze_positions

                    frame_id = 0
                    prev_gaze_positions = []
                    #null_values = []
                    for line in full_data:
                        line = line.split(',')
                        frame_id = int(line[0].split('_')[2])
                        episode_id = line[1]
                        score = line[2]
                        duration = line[3]
                        unclipped_reward = line[4]
                        action = line[5]
                        gaze_positions = line[6:]
                        if gaze_positions[0]!= 'null':
                            prev_gaze_positions = gaze_positions
                            saliency_map = create_gaussian_map(gaze_positions)
                        else:
                            #if positions is Null
                            #empty_img = np.zeros((210, 160, 3), np.uint8)
                            #saliency_map = cv2.cvtColor(empty_img, cv2.COLOR_BGR2GRAY)
                            if len(prev_gaze_positions)==0:
                                print("Error: could not interpolate " + str(frame_id), flush=True)
                                return
                            saliency_map = create_gaussian_map(prev_gaze_positions)
                        if not cv2.imwrite(os.path.join(path_annt, '%06d.png' %(frame_id)), saliency_map):
                            print("could not write image!", flush=True)
                    print(frame_id)
                    number_of_frames.append(frame_id)

                    #interpolate null values
                    #interpolate_null_values(full_data, null_values, path_annt, frame_id)
    print(number_of_frames)
    if os.path.isfile(num_frame_path):
        os.remove(num_frame_path)
    with open(num_frame_path, "w", newline = '') as f:
        writer = csv.writer(f)
        for element in number_of_frames:
            writer.writerow([element])

def create_gaussian_map(positions):
    x = np.array(positions[::2])
    y = np.array(positions[1::2])
    x = (x.astype(np.float)).astype(np.int)
    y = (y.astype(np.float)).astype(np.int)
    if len(x) != len(y):
        print("Error: Length of x and y vary")
    img = np.zeros((210, 160, 3), np.uint8)
    for i in range(len(x)):
        x_temp = x[i]
        y_temp = y[i]
        if (x_temp >= 160): x_temp = 159
            #print("overflow x by :" + str(x_temp - 159))
        if (x_temp < 0): x_temp = 0
        if (y_temp >= 210): y_temp = 209
            #print("overflow y :" + str(y_temp - 210))
        if (y_temp < 0): y_temp = 0

        cv2.circle(img, (x_temp, y_temp), 15, (255, 255, 255), -1)
        #print(str(x[i]) + ", " + str(y[i]))
    blurred_img = cv2.cvtColor(cv2.GaussianBlur(img, (45, 45), 0), cv2.COLOR_BGR2GRAY)
    return blurred_img
    #TODO: check for size of original picture, correct interpolation

def interpolate_null_values(full_data, null_values, path_annt, num_frame):
    for id in null_values:
        imgs = []
        if (id==num_frame):
            temp_id = id
            flag = 0;
            while not flag:
                temp_id -= 1
                if os.path.isfile(os.path.join(path_annt, '%06d.png' %temp_id)):
                    flag = 1
                    imgs.append(cv2.imread(os.path.join(path_annt, '%06d.png' %temp_id)))

        elif (id==1):
            temp_id = id
            flag = 0;
            while not flag:
                temp_id += 1
                if os.path.isfile(os.path.join(path_annt, '%06d.png' % temp_id)):
                    flag = 1
                    imgs.append(cv2.imread(os.path.join(path_annt, '%06d.png' % temp_id)))
        else:
            while not flag:
                temp_id += 1
                if os.path.isfile(os.path.join(path_annt, '%06d.png' % temp_id)):
                    flag = 1
                    imgs.append(cv2.imread(os.path.join(path_annt, '%06d.png' % temp_id)))
                    #.....
        new_map = np.zeros(shape=[384, 224, 3], dtype=np.uint8)

        cv2.imwrite(os.path.join(path_annt, '%06d.png' % id), new_map)

# adaption from gist.github.com/keithweaver/70df4922fec74ea87405b83840b45d57
def split_video_to_frames(path_indata):
    FPS = 25
    path_video = os.path.join(path_indata, 'video')
    for file in os.listdir(path_video):
        if file.endswith("AVI"):
            index = int(file.split('.')[0])
            new_path = os.path.join(path_video, '%04d'%(index))
            # Playing video from file: 384, 224
            cap = cv2.VideoCapture(os.path.join(path_video, file))
            cap.set(cv2.CAP_PROP_FPS, FPS)

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
                file_name = '%06d.png' % (currentFrame + 1)
                name = os.path.join(new_path, file_name)
                # print ('Creating... ' + name)
                frame = cv2.resize(frame, (384, 224))
                cv2.imwrite(name, frame)
                currentFrame += 1

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()

            #rename files to %06d
            path_annt = os.path.join(path_indata, 'annotation', '%04d'%(index), 'maps')
            for frame in os.listdir(path_annt):
                os.rename(os.path.join(path_annt, frame), os.path.join(path_annt, '%06d' %int(frame.split('.')[0]) + '.png'))
            print('splitted: ' + str(index))

    return 0


def main():
    ''' preprocessing of input data '''
    # split into frames if needed
    # read in atari data and save accordingly
    yes = set(['yes', 'y'])
    no = set(['no', 'n', ''])
    answered = False
    while not answered:
        answer = input("Preprocess Atari data? (y/n) ").lower()
        answered = True
        if answer in yes:
            atari_reader('Atari_dataset')
    answered = False
    while not answered:
        answer = input("Split frames of DHF1K data? (y/n) ").lower()
        answered = True
        if answer in yes:
            split_video_to_frames('DHF1K_dataset')



if __name__ == '__main__':
    main()
