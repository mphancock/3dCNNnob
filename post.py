import numpy as np

import os

from scipy.misc import imsave

from predict import process_probability_map

from clear_directories import clear_data_directories
from clear_directories import clear_directory

from save_data import save_data

from cost_functions import dice_coeff

from utilities import roi_overlay

'''
    preview 
        --> roi_pred 
        --> roi 
        --> image 
        --> true_overlay
        --> pred_overlay 
        --> affine 
'''

def get_data(data_path, file):
    directories = ['train', 'test', 'val']
    for dir in directories:
        sub_dir = os.path.join(data_path, dir, 'image')
        files = os.listdir(sub_dir)

        for i in files:
            if i == file:
                file_path = os.path.join(sub_dir, i)
                roi_file_path = os.path.join(data_path, dir, 'roi', i)

                img = np.load(file_path)
                roi = np.load(roi_file_path)

                print('found file {} in {} directory'.format(file, dir))
                return img, roi


def check_pred(pred_roi, num):
    for i in np.nditer(pred_roi):
        if i != 1 and i != 0:
            print('{} has non zero/one array element {}'.format(num, i))


def clear_save_directories(save_dir):
    dir_list = ['roi_pred', 'roi', 'image', 'true_overlay', 'pred_overlay', 'affine']
    for dir in dir_list:
        dir = os.path.join(save_dir, dir)
        clear_directory(dir)


if __name__ == '__main__':
    data_path = '/Users/Matthew/Documents/Research/3dData/'

    mat_path = '/Users/Matthew/Documents/Research/corrMTRdata'

    clear_data_directories(data_path)

    save_data(mat_path, data_path)


    pred_dir = '/Users/Matthew/Downloads/pred'
    pred_files = os.listdir(pred_dir)

    if '.DS_Store' in pred_files:
        pred_files.remove('.DS_Store')

    save_dir = '/Users/Matthew/Documents/Research/preview'
    clear_save_directories(save_dir)

    for file in pred_files:
        file_path = os.path.join(pred_dir, file)
        pred_roi = np.load(file_path)
        file_num = file[0:5]

        check_pred(pred_roi, file_num)

        pred_roi = process_probability_map(pred_roi)
        img, roi = get_data(data_path, file)

        for i in range(pred_roi.shape[2]):
            file_name = '{}slice{}.jpg'.format(file_num, i)
            file_name = os.path.join(save_dir, 'roi_pred', file_name)
            imsave(file_name, pred_roi[:, :, i])

        for i in range(img.shape[2]):
            img_file_name = '{}slice{}.jpg'.format(file_num, i)
            img_file_name = os.path.join(save_dir, 'image', img_file_name)
            imsave(img_file_name, img[:, :, i])

        for i in range(roi.shape[2]):
            roi_file_name = '{}slice{}.jpg'.format(file_num, i)
            roi_file_name = os.path.join(save_dir, 'roi', roi_file_name)
            imsave(roi_file_name, roi[:, :, i])

        rgb_true = roi_overlay(img, roi, img.shape)
        rgb_pred = roi_overlay(img, pred_roi, img.shape)

        for i in range(rgb_true.shape[0]):
            true_file_name = '{}slice{}.jpg'.format(file_num, i)
            true_file_name = os.path.join(save_dir, 'true_overlay', true_file_name)
            imsave(true_file_name, rgb_true[i, :, :, :])

        for i in range(rgb_pred.shape[0]):
            pred_file_name = '{}slice{}.jpg'.format(file_num, i)
            pred_file_name = os.path.join(save_dir, 'pred_overlay', pred_file_name)
            imsave(pred_file_name, rgb_pred[i, :, :, :])

        dc = dice_coeff(roi, pred_roi)
        print('{} prediction has been saved'.format(file_num))
        print('{} image has been saved'.format(file_num))
        print('{} pred roi dice coefficient:\t{}'.format(file_num, dc))

    print('all files saved')


