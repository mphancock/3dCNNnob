import numpy as np
import os

import scipy.io as sio
# from scipy.misc import imsave
from subprocess import call

from predict import process_probability_map

from utilities import clear_dir, roi_overlay

from data import normalize_image

from matplotlib.image import imsave
import matplotlib.pyplot as plt


# def get_data(data_path, file):
#     directories = ['train', 'test', 'val']
#     for dir in directories:
#         sub_dir = os.path.join(data_path, dir, 'image')
#         files = os.listdir(sub_dir)
#
#         for i in files:
#             if i == file:
#                 file_path = os.path.join(sub_dir, i)
#                 roi_file_path = os.path.join(data_path, dir, 'roi', i)
#
#                 img = np.load(file_path)
#                 roi = np.load(roi_file_path)
#
#                 print('found file {} in {} directory'.format(file, dir))
#                 return img, roi


def check_pred(pred_roi, num):
    for i in np.nditer(pred_roi):
        if i != 1 and i != 0:
            print('{} has non zero/one array element'.format(num))


def dice_coeff(r1, r2):
    true_f = r1.flatten()
    pred_f = r2.flatten()
    intersect = np.sum(true_f*pred_f)
    return (2 * intersect) / (np.sum(true_f) + np.sum(pred_f))


def save_3d(save_img, dir_path, file_num):
    for i in range(save_img.shape[2]):
        file_name = '{}slice{}.jpg'.format(file_num, i)
        file_path = os.path.join(dir_path, file_name)
        slice = save_img[:, :, i]
        f_slice = np.flipud(slice)

        for i in range(3):
            f_slice = np.rot90(f_slice)

        # imsave(file_path, f_slice)
        imsave(file_path, f_slice)


def save_from_cloud(base_path):
    cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/trial/'
    local_dir = base_path
    call(['scp', '-r', cloud_dir, local_dir])


def mask_processing(img, num, mat_path):
    mat_files = os.listdir(mat_path)

    if '.DS_Store' in mat_files:
        mat_files.remove('.DS_Store')

    file = ''
    for i in mat_files:
        if num in i:
            file = i
            break

    if file == '':
        raise Exception('Could not find corresponding mat file to: '.format(num))

    mat_dict = sio.loadmat(os.path.join(mat_path, file))
    mask = mat_dict['bgMASK']

    mask = normalize_image(mask)
    if(mask.shape != (256, 256, 40)):
        print('\t\t{} incorrect mask shape: {}'.format(num, mask.shape))
        return img

    mask_img = np.multiply(img, mask)

    return mask_img


def gen_net_prediction(trial_path, net):
    mat_dir = 'corrMTRdata'
    mat_path = os.path.join('/Users/Matthew/Documents/Research', mat_dir)
    net_path = os.path.join(trial_path, net)
    data_path = os.path.join(trial_path, '3dData')

    pred_dir = os.path.join(net_path, 'pred')
    pred_files = os.listdir(pred_dir)

    if '.DS_Store' in pred_files:
        pred_files.remove('.DS_Store')

    prev_path = os.path.join(net_path, 'preview')
    succ_ave_dc = 0
    over_ave_dc = 0
    cmt_ave_dc = 0

    fail = 0
    num_cmt = 0

    for file in pred_files:
        pred_roi = np.load(os.path.join(pred_dir, file))
        file_num = file[0:5]

        # pred_roi = process_probability_map(pred_roi)
        check_pred(pred_roi, file_num)

        pred_roi = mask_processing(img=pred_roi, num=file_num, mat_path=mat_path)

        img = np.load(os.path.join(data_path, 'test', 'image', file))
        roi = np.load(os.path.join(data_path, 'test', 'roi', file))

        save_3d(pred_roi, os.path.join(prev_path, 'roi_pred'), file_num)
        save_3d(img, os.path.join(prev_path, 'image'), file_num)

        rgb_true = roi_overlay(img, roi)
        rgb_pred = roi_overlay(img, pred_roi)

        rgb_true = rgb_true.transpose(1, 2, 0, 3)
        rgb_pred = rgb_pred.transpose(1, 2, 0, 3)

        save_3d(rgb_true, os.path.join(prev_path, 'true_overlay'), file_num)
        save_3d(rgb_pred, os.path.join(prev_path, 'pred_overlay'), file_num)

        dc = dice_coeff(roi, pred_roi)
        print('\t\t{} pred roi dice coefficient:\t{}'.format(file_num, dc))
        if dc > .5:
            succ_ave_dc += dc
        else:
            fail += 1
            print('\t\t\tSegmentation failed')

        over_ave_dc += dc

        if file_num[0] == '2':
            num_cmt += 1
            cmt_ave_dc += dc

    succ_ave_dc = succ_ave_dc / (len(pred_files)-fail)
    over_ave_dc = over_ave_dc / (len(pred_files))
    cmt_ave_dc = cmt_ave_dc / num_cmt

    return succ_ave_dc, over_ave_dc, cmt_ave_dc


def gen_trial_predictions(base_path, trial):
    print('Trial {} Predictions: '.format(trial))
    trial_path = os.path.join(base_path, 'trial', 'trial{}'.format(trial))

    for net in ['cnn2d', 'cnn3d', 'cnnbn', 'cnnwo']:
        print('\tNet: {}'.format(net))
        net_dc = gen_net_prediction(trial_path, net)
        print('\t{} average succesfull predictive dice coefficient:\t{}'.format(net, net_dc[0]))
        print('\t{} average overall predictive dice coefficient:\t{}'.format(net, net_dc[1]))
        print('\t{} average CMT predictive dice coefficient:\t\t{}'.format(net, net_dc[2]))

    print('Trial {} Post-processing complete\n'.format(trial))


if __name__ == '__main__':
    base_path = '/Users/Matthew/Documents/Research/'
    #
    # if(os.path.isdir(os.path.join(base_path, 'trial'))):
    #     clear_dir(os.path.join(base_path, 'trial'))

    # save_from_cloud(base_path)

    clear_dir(os.path.join(base_path, 'trial', 'trial1', 'cnn2d', 'preview'))
    clear_dir(os.path.join(base_path, 'trial', 'trial1', 'cnn3d', 'preview'))


    for i in range(1,4):
        trial = i
        gen_trial_predictions(base_path, trial)

    print('fin')


