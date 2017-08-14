import numpy as np

import os

from scipy.misc import imsave
from subprocess import call

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


def save_pred(save_img, save_path, file_num):
    for i in range(save_img.shape[2]):
        file_name = '{}slice{}.jpg'.format(file_num, i)
        file_path = os.path.join(save_path, file_name)
        imsave(file_path, save_img[:, :, i])


def gen_predictions(thresh):
    data_path = '/Users/Matthew/Documents/Research/3dData/'
    mat_path = '/Users/Matthew/Documents/Research/corrMTRdata'

    clear_data_directories(data_path)

    cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/3dData'
    local_dir = '/Users/Matthew/Documents/Research/'
    call(['scp', '-r', cloud_dir, local_dir])

    save_data(mat_path, data_path)

    pred_dir = '/Users/Matthew/Downloads/pred'
    pred_files = os.listdir(pred_dir)

    aff_dir = '/Users/Matthew/Downloads/aff'
    aff_files = os.listdir(aff_dir)

    if '.DS_Store' in pred_files:
        pred_files.remove('.DS_Store')

    if '.DS_Store' in aff_files:
        aff_files.remove('.DS_Store')

    save_dir = '/Users/Matthew/Documents/Research/preview'
    clear_save_directories(save_dir)

    ave_dc = 0
    ave_successful_dc = 0
    successful_reg = []
    ave_unsuccessful_dc = 0
    unsuccesful_reg = []

    for file in pred_files:
        file_path = os.path.join(pred_dir, file)
        pred_roi = np.load(file_path)
        file_num = file[0:5]

        check_pred(pred_roi, file_num)

        pred_roi = process_probability_map(pred_roi)
        img, roi = get_data(data_path, file)

        save_pred(pred_roi, os.path.join(save_dir, 'roi_pred'), file_num)
        save_pred(img, os.path.join(save_dir, 'image'), file_num)
        save_pred(roi, os.path.join(save_dir, 'roi'), file_num)

        rgb_true = roi_overlay(img, roi, img.shape)
        rgb_pred = roi_overlay(img, pred_roi, img.shape)

        save_pred(rgb_true, os.path.join(save_dir, 'true_overlay'), file_num)
        save_pred(rgb_pred, os.path.join(save_dir, 'pred_overlay'), file_num)

        dc = dice_coeff(roi, pred_roi)
        ave_dc += dc

        if dc > thresh:
            ave_successful_dc += dc
            successful_reg.append(file_num)
        else:
            ave_unsuccessful_dc += dc
            unsuccesful_reg.append(file_num)

        print('{} pred roi dice coefficient:\t{}'.format(file_num, dc))

    for file in aff_files:
        file_path = os.path.join(aff_dir, file)
        aff = np.load(file_path)
        file_num = file[0:5]

        save_pred(aff, os.path.join(save_dir, 'affine'), file_num)

    ave_dc = ave_dc / len(pred_files)
    ave_successful_dc = ave_successful_dc / len(successful_reg)
    ave_unsuccessful_dc = ave_unsuccessful_dc / len(unsuccesful_reg)

    print('overall average dice coefficient: {}\n'.format(ave_dc))
    print('successful registration: {}'.format(successful_reg))
    print('successful registration average dice coefficient: {}'.format(ave_successful_dc))
    print('unsuccesful registration: {}'.format(unsuccesful_reg))
    print('unsuccesful registration average dice coefficient: {}'.format(ave_unsuccessful_dc))

    print('all files saved')


if __name__ == '__main__':
    tresh = .1
    gen_predictions(tresh)
    print('fin')


