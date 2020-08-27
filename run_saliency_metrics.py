#from: https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py

import numpy as np
import os
import cv2
import random
import csv
from tqdm import tqdm
from model import TASED_v2
import torch

from scipy.ndimage import gaussian_filter

#import inferencer_model
#import inference_controller

def main():
    shuffle_data_beforehand = True
    weight_file = 'space_invaders_weights_1002.pt' #'produced_weight_file.pt' on Server
    num_iters = 100
    calculate_shuff = True
    #save_inference_files = True
    name = "SpaceInvaders Metrics: %d iterations" %num_iters
    save_errors = False

    tuned_weights = os.path.join('Atari_dataset', weight_file)
    path_annt = os.path.join('Atari_dataset', 'annotation')
    path_frames = os.path.join('Atari_dataset', 'video')
    path_output = os.path.join('output', 'temp')

    #path_tuned_smap = ''
    #path_original_smap = ''

    model_original = build_sal_model('TASED_updated.pt') #  'space_invaders_weights_1001.pt') #
    model_finetuned = build_sal_model(tuned_weights)

    #model_original = inferencer_model.Inferencer_Model()
    #controller_original = inference_controller.Inferencer_Controller(model_original)
    #model_original.file_weight = 'TASED_updated.pt'
    #controller_original.prepare_model()

    #model_finetuned = inferencer_model.Inferencer_Model()
    #controller_finetuned = inference_controller.Inferencer_Controller(model_finetuned)
    #model_finetuned.file_weight = os.path.join('Atari_dataset', 'enduro_weights_3000.pt')
    #controller_finetuned.prepare_model()

    scores = {}
    scores['judd_t'] = []
    scores['judd'] = []
    scores['shuff_t'] = []
    scores['shuff'] = []
    scores['sim_t'] = []
    scores['sim'] = []

    if shuffle_data_beforehand:
        split_train_test_set()

    csv_reader = csv.reader(open('Atari_num_frame_testing.csv', 'r'))
    list_of_tuples = list(map(tuple, csv_reader))  # list of (#samples, file_name)
    length_array = [int(i[0]) for i in list_of_tuples]
    index_array = [i[1] for i in list_of_tuples]

    #for i in tqdm(range(num_iters)):
    i = 0
    error_counter = 0
    while i < num_iters:
        print("[%d/%d]" %(i, num_iters))
        # file, frame_number = get_random_sample(length_array)

        file, frame_number, images = get_random_clip(length_array, index_array, path_frames)

        #
        """for frame in images:
            cv2.imshow('debug', frame)
            cv2.waitKey()"""

        #model_original.images_dict[0] = images
        #model_finetuned.images_dict[0] = images

        #controller_original.inferencer.process_all()
        #controller_finetuned.inferencer.process_all()

        #original_smap = model_original.current_saliency_map_dict[0][:, :, 0]
        #tuned_smap = model_finetuned.current_saliency_map_dict[0][:, :, 0]

        original_smap = produce_saliency_map(images, model_original)
        tuned_smap = produce_saliency_map(images, model_finetuned)

        if np.max(original_smap) > 0:
            i += 1
        else:
            error_counter += 1
            if save_errors:
                path_errors = os.path.join(path_output, str(error_counter))
                if not os.path.isdir(path_errors):
                    os.makedirs(path_errors)
                for j in range(len(images)):
                    cv2.imwrite(os.path.join(path_errors, '%04d.png' %(j)), images[j])
                cv2.imwrite(os.path.join(path_errors, 'smap_original.png'), original_smap)
                cv2.imwrite(os.path.join(path_errors, 'smap_tuned.png'), tuned_smap)
            continue
        #print("original" + str(np.max(original_smap)))
        #print("Tuned" + str(np.max(tuned_smap)))
        #
        # cv2.imshow('with tuned model', tuned_smap)
        # cv2.waitKey()
        #
        # cv2.imshow('with original model', original_smap)
        # cv2.waitKey()

        gt = cv2.imread(os.path.join(path_annt, file, 'discrete', '%06d.png'%(frame_number)), cv2.IMREAD_GRAYSCALE)

        cv2.imwrite(os.path.join(path_output, '%06d_01orig_%06d.png' % (i, frame_number)), original_smap)
        cv2.imwrite(os.path.join(path_output, '%06d_02tuned%06d.png' % (i, frame_number)), tuned_smap)
        cv2.imwrite(os.path.join(path_output, '%06d_03input%06d.png' % (i, frame_number)), images[-1])

        # print(os.path.join(path_annt, file, 'discrete', '%06d.png'%(frame_number)))

        #cv2.imshow('debug', gt)
        #cv2.waitKey()

        #tuned_smap = cv2.imread(os.path.join(path_output, '%06d_02tuned%06d.png' % (i, frame_number)), cv2.IMREAD_GRAYSCALE)
        #original_smap = cv2.imread(os.path.join(path_output, '%06d_01orig_%06d.png' % (i, frame_number)), cv2.IMREAD_GRAYSCALE)

        scores['judd'].append(auc_judd(original_smap, gt))
        scores['judd_t'].append(auc_judd(tuned_smap, gt))

        scores['sim'].append(similarity(original_smap, gt))
        scores['sim_t'].append(similarity(tuned_smap, gt))

        if calculate_shuff:
            normalize_properly = False
            while not normalize_properly:
                file, frame_number, images = get_random_clip(length_array, index_array, path_frames)

                #model_original.images_dict[0] = images
                #model_finetuned.images_dict[0] = images

                #controller_original.inferencer.process_all()
                #controller_finetuned.inferencer.process_all()

                #original_smap_other = model_original.current_saliency_map_dict[0][:, :, 0]
                #tuned_smap_other = model_finetuned.current_saliency_map_dict[0][:, :, 0]

                original_smap_other = produce_saliency_map(images, model_original)
                tuned_smap_other = produce_saliency_map(images, model_finetuned)

                if np.max(original_smap_other) > 0:
                    normalize_properly = True
                else:
                    error_counter += 1
                    continue

                scores['shuff_t'].append(auc_shuff_acl(tuned_smap, gt, tuned_smap_other))
                scores['shuff'].append((auc_shuff_acl(original_smap, gt, original_smap_other)))

    judd_average_tuned = sum(scores['judd_t'])/len(scores['judd_t'])
    judd_average_original = sum(scores['judd'])/len(scores['judd'])

    sim_average_tuned = sum(scores['sim_t']) / len(scores['sim_t'])
    sim_average_original = sum(scores['sim']) / len(scores['sim'])

    shuff_average_tuned = 0
    shuff_average_original = 0

    if calculate_shuff:
        shuff_average_tuned = sum(scores['shuff_t'])/len(scores['shuff_t'])
        shuff_average_original = sum(scores['shuff'])/len(scores['shuff'])


    print(name + " (%d zero valued original_smap's)" %(error_counter))
    print('original (AUC_Judd, AUC_shuff, similarity): ' + str((judd_average_original, shuff_average_original, sim_average_original)))
    print('tuned (AUC_Judd, AUC_shuff, similarity): ' + str((judd_average_tuned, shuff_average_tuned, sim_average_tuned)))

def get_random_sample(length_array):
    frame = random.randint(0, sum(length_array))
    file = 0
    while frame > length_array[file]:
        frame -= length_array[file]
        file += 1
    file += 1
    frame += 1
    return file, frame

def get_random_clip(length_array, index_array, path_frames):
    rand_file = random.randint(0, len(length_array) - 1)

    start_idx = random.randint(32, length_array[rand_file] - 1 - 32)
    file_name = index_array[rand_file]


    clip = []
    for i in range(32):
        image_path = os.path.join(path_frames, file_name, '%06d.png' % (start_idx + i + 1))
        # print(os.path.abspath(image_path))
        img = cv2.imread(image_path)
        clip.append(img)

    return file_name, start_idx + 32, clip

def produce_saliency_map(snippet, sal_model):
    clip = transform(snippet)
    smap = process(sal_model, clip)
    # shape of smap: (210, 160, 1) np array
    # print(np.max(smap))

    return smap

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

def process(model, clip):
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.cuda()).cpu().data[0]
    smap = (smap.numpy()*255.).astype(np.int)/255.
    #print(smap)
    smap = gaussian_filter(smap, sigma=7)
    smap = cv2.resize(smap, (160, 210))

    return (smap/np.max(smap)*255.).astype(np.uint8) #error if smap is zero everywhere

def build_sal_model(file_weight):
    model = TASED_v2()

    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(' size? ' + name, param.size(), model_dict[name].size())
            else:
                print(' name? ' + name)

        print(' loaded')
    else:
        print('weight file?')

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    return model

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    min = np.min(s_map)
    max = np.max(s_map)

    #print(s_map.shape)
    #cv2.imshow('debug', s_map)
    #cv2.waitKey()

    assert max > 0.0,\
        'Error in normalization. max value not larger than 0'

    norm_s_map = (s_map - min) / (max - min)

    # min_n = np.min(norm_s_map)
    # max_n = np.max(norm_s_map)

    return norm_s_map


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map_norm = normalize_map(s_map)
    #print(np.max(s_map_norm))
    gt_norm = normalize_map(gt)
    assert np.max(gt_norm) == 1.0,\
        'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map_norm) == 1.0,\
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map_norm[gt_norm > 0].tolist()

    num_fixations = len(thresholds)
    # num fixations is no. of salience map values at gt >0

    thresholds = sorted(set(thresholds))

    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map,
        # keep only those pixels with values above threshold
        temp = s_map >= thresh
        num_overlap = np.sum(np.logical_and(temp, gt))
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap
        # with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / (np.prod(gt.shape[:2]) - num_fixations)

        area.append((round(tp, 4) ,round(fp, 4)))

    area.append((1.0, 1.0))
    area.sort(key=lambda x: x[0])
    tp_list, fp_list = list(zip(*area))
    return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_shuff_acl(s_map, gt, other_map, n_splits=100, stepsize=0.1):

    # If there are no fixations to predict, return NaN
    if np.sum(gt) == 0:
        print('no gt')
        return None

    # normalize saliency map
    s_map = normalize_map(s_map)

    S = s_map.flatten()
    F = gt.flatten()
    Oth = other_map.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)

    # for each fixation, sample Nsplits values from the sal map at locations
    # specified by other_map

    ind = np.where(Oth > 0)[0]  # find fixation locations on other images

    Nfixations_oth = min(Nfixations, len(ind))
    randfix = np.full((Nfixations_oth, n_splits), np.nan)

    for i in range(n_splits):
        # randomize choice of fixation locations
        randind = np.random.permutation(ind.copy())
        # sal map values at random fixation locations of other random images
        randfix[:, i] = S[randind[:Nfixations_oth]]

    # calculate AUC per random split (set of random locations)
    auc = np.full(n_splits, np.nan)
    for s in range(n_splits):

        curfix = randfix[:, s]

        allthreshes = np.flip(np.arange(0, max(np.max(Sth), np.max(curfix)), stepsize))
        tp = np.zeros(len(allthreshes) + 2)
        fp = np.zeros(len(allthreshes) + 2)
        tp[-1] = 1
        fp[-1] = 1

        for i in range(len(allthreshes)):
            thresh = allthreshes[i]
            tp[i + 1] = np.sum(Sth >= thresh) / Nfixations
            fp[i + 1] = np.sum(curfix >= thresh) / Nfixations_oth

        auc[s] = np.trapz(np.array(tp), np.array(fp))

    return np.mean(auc)

def similarity(s_map, gt):
    s_map_norm = normalize_map(s_map)
    gt_norm = normalize_map(gt)
    return np.sum(np.minimum(s_map_norm, gt_norm))

def split_train_test_set():
    path_full = 'Atari_num_frame_FullData.csv'
    path_train = 'Atari_num_frame_train.csv'
    path_test = 'Atari_num_frame_testing.csv'

    list_num_frame = [int(row[0]) for row in csv.reader(open(path_full, 'r'))]
    total_len = len(list_num_frame)
    half_len = int(np.floor(total_len/2)) #for 50/50 split
    idx = range(1, total_len+1)
    z = list(zip(list_num_frame, idx))

    #comment this out for training on ~ highscore data
    random.shuffle(z)

    list_num_frame, idx = zip(*z)

    test_list, test_idx = list_num_frame[:half_len], idx[:half_len]
    train_list, train_idx = list_num_frame[half_len:], idx[half_len:]
    train_strings = []
    test_strings = []

    for i in range(len(train_list)):
        train_strings.append(str(train_list[i])+',%04d'%train_idx[i])
    for i in range(len(test_list)):
        test_strings.append(str(test_list[i])+',%04d'%test_idx[i])

    with open(path_train, 'w') as file:
        for line in train_strings:
            file.write(line)
            file.write('\n')
    with open(path_test, 'w') as file:
        for line in test_strings:
            file.write(line)
            file.write('\n')

    return 0

if __name__ == '__main__':
    main()