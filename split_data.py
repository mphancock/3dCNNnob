import os

import scipy.io as sio
import numpy as np

from random import shuffle

from data import zero_pad
from data import normalize_image


def split_data(mat_path, data_path, split_pct=.8):
    files = os.listdir(mat_path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    print('original number of files: {}'.format(len(files)))
    #removes duplicate file name
    files = list(set(files))
    shuffle(files)

    num_files = len(files)
    print('abridged number of files: {}'.format(num_files))

    num_test = int(num_files * (1 - split_pct))
    num_val = int((num_files - num_test) * (1-split_pct))

    count = 0
    for file in files:
        if count < num_test:
            path = os.path.join(data_path, 'test')
        elif count < num_test + num_val:
            path = os.path.join(data_path, 'val')
        else:
            path = os.path.join(data_path, 'train')
        process_file(file, mat_path, path)

        count += 1

    print('All .MAT data saved to the following directory: {}'.format(data_path))


def process_file(file, mat_path, data_path):
    mat_dict = sio.loadmat(os.path.join(mat_path, file))
    image = mat_dict['MTwcorr']
    roi = mat_dict['nerveROI']

    for i in range(roi.shape[2]):
        if np.sum(roi[:, :, i]) == 0:
            print('\nThe following expert rated roi is incomplete: {}\n'.format(file))
            return

    name = file[9:14]

    hit = False

    # python 2.7 syntax: range(x,y)
    # python 3.5 syntax: list(range(x,y))
    if file[14] in [str(n) for n in range(0,10)]:
        name = file[9:16]
        hit = True

    shape = (256, 256, 40)

    if image.shape[0] < shape[0]:
        image = zero_pad(image, shape)
        roi = zero_pad(roi, shape)

    if image.shape == shape and roi.shape == shape:
        np.save(os.path.join(data_path, 'image', name), normalize_image(image))
        np.save(os.path.join(data_path, 'roi', name), normalize_image(roi))
        print('ndarray saved -- name: {}\tshape: {}\tpath: {}\thit: {}'.format(name, image.shape, data_path, hit))
    else:
        print('{} invalid dimensions:\nimage dimension:{}\nroi dimension:{}'.format(name, image.shape, roi.shape))





