import os

import numpy as np

def get_count(path):
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    return len(files)


def sum_border(roi):
    sum = 0
    for i in range(roi.shape[2]):
        slice = roi[:, :, i]
        for j in range(roi.shape[0]):
            row = slice[j, :]
            if j == 0 or j == (roi.shape[0]-1):
                sum += np.sum(row)
            else:
                for k in range(roi.shape[1]):
                    if k == 0 or k == (roi.shape[1]-1):
                        sum += roi[j, k, i]
    return sum


def roi_overlay(image, roi, shape):
    rgb = np.ndarray([shape[2], shape[0], shape[1], 3])

    img_t = image.T
    roi_t = roi.T

    rgb[:, :, :, 0] = img_t

    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for k in range(rgb.shape[2]):
                if roi_t[i, j, k] > .5:
                    rgb[i, j, k, 0] = 1

    rgb[:, :, :, 1] = img_t
    rgb[:, :, :, 2] = img_t

    return rgb










