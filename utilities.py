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


