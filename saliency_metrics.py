#from: https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py

import numpy as np
import os
import cv2
import random
import csv

#first run create_maps.py to produce saliency maps with finetuned weights
def main():
    num_iters = 1000

    path_annt = os.path.join('Atari_dataset', 'annotation')
    path_tuned_smap = os.path.join('output', 'finetuned')
    path_original_smap = os.path.join('output', 'TASED_original')

    scores = {}
    scores['judd_t'] = []
    scores['judd'] = []
    scores['shuff_t'] = []
    scores['shuff'] = []
    scores['sim_t'] = []
    scores['sim'] = []

    length_array = [int(row[0]) for row in csv.reader(open('Atari_num_frame_train.csv', 'r'))]
    length_array = length_array[0:18]

    for i in range(num_iters):
        file, frame_number = get_random_sample(length_array)
        gt = cv2.imread(os.path.join(path_annt, '%04d'%(file), 'discrete', '%06d.png'%(frame_number)), cv2.IMREAD_GRAYSCALE)
        tuned_smap = cv2.imread(os.path.join(path_tuned_smap, '%04d'%(file), '%06d.png'%(frame_number)), cv2.IMREAD_GRAYSCALE)
        original_smap = cv2.imread(os.path.join(path_original_smap, '%04d'%(file), '%06d.png'%(frame_number)), cv2.IMREAD_GRAYSCALE)
        file, frame_number = get_random_sample(length_array)
        other_smap = cv2.imread(os.path.join(path_tuned_smap, '%04d'%(file), '%06d.png'%(frame_number)), cv2.IMREAD_GRAYSCALE)
        other_original_smap = cv2.imread(os.path.join(path_original_smap, '%04d'%(file), '%06d.png'%(frame_number)), cv2.IMREAD_GRAYSCALE)
        scores['judd_t'].append(auc_judd(tuned_smap, gt))
        scores['judd'].append(auc_judd(original_smap, gt))
        scores['shuff_t'].append(auc_shuff_acl(tuned_smap, gt, other_smap))
        scores['shuff'].append((auc_shuff_acl(original_smap, gt, other_original_smap)))
        scores['sim_t'].append(similarity(tuned_smap, gt))
        scores['sim'].append(similarity(original_smap, gt))

    judd_average_tuned = sum(scores['judd_t'])/len(scores['judd_t'])
    judd_average_original = sum(scores['judd'])/len(scores['judd'])
    shuff_average_tuned = sum(scores['shuff_t'])/len(scores['shuff_t'])
    shuff_average_original = sum(scores['shuff'])/len(scores['shuff'])
    sim_average_tuned = sum(scores['sim_t']) / len(scores['sim_t'])
    sim_average_original = sum(scores['sim']) / len(scores['sim'])
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

def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map


def auc_judd(s_map, gt):
    # ground truth is discrete, s_map is continous and normalized
    s_map = normalize_map(s_map)
    assert np.max(gt) == 1.0,\
        'Ground truth not discretized properly max value > 1.0'
    assert np.max(s_map) == 1.0,\
        'Salience map not normalized properly max value > 1.0'

    # thresholds are calculated from the salience map,
    # only at places where fixations are present
    thresholds = s_map[gt > 0].tolist()

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
    return np.sum(np.minimum(s_map, gt))

if __name__ == '__main__':
    main()