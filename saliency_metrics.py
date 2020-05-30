#from: https://github.com/rdroste/unisal/blob/master/unisal/salience_metrics.py

import numpy as np
import os
import cv2

#first run run_visualize.py to produce saliency maps with finetuned weights
def main():
    path_annt = os.path.join('Atari_dataset', 'annotation')
    path_gt = os.path.join(path_annt, '0001', 'discrete')
    path_tuned_smap = os.path.join('output', 'finetuned', '0001')
    path_original_smap = os.path.join('output', 'TASED_original', '0001')
    list_frames = [d for d in os.listdir(path_original_smap) if os.path.isfile(os.path.join(path_tuned_smap, d))]
    list_frames.sort()

    judd_tuned = []
    judd_original = []
    for i in range(len(list_frames)):
        gt_frame = cv2.imread(os.path.join(path_gt, list_frames[i]), cv2.IMREAD_GRAYSCALE)
        smap_frame = cv2.imread(os.path.join(path_tuned_smap, list_frames[i]), cv2.IMREAD_GRAYSCALE)
        judd_tuned.append(auc_judd(smap_frame, gt_frame))
        smap_frame = cv2.imread(os.path.join(path_original_smap, list_frames[i]), cv2.IMREAD_GRAYSCALE)
        judd_original.append(auc_judd(smap_frame, gt_frame))

    average_tuned = sum(judd_tuned)/len(judd_tuned)
    average_original = sum(judd_original)/len(judd_original)
    print('original: ' + str(average_original))
    print('tuned: ' + str(average_tuned))

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