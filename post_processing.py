import numpy as np
import os

from scipy.misc import imsave
from subprocess import call

from predict import process_probability_map

from utilities import clear_dir, roi_overlay

from cost_functions import dice_coeff

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


def save_pred(save_img, dir_path, file_num):
    for i in range(save_img.shape[2]):
        file_name = '{}slice{}.jpg'.format(file_num, i)
        file_path = os.path.join(dir_path, file_name)
        imsave(file_path, save_img[:, :, i])


def save_from_cloud(trial):
    cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/trial{}/'.format(trial)
    local_dir = '/Users/Matthew/Documents/Research/'
    call(['scp', '-r', cloud_dir, local_dir])


def gen_net_prediction(base_path, net):
    net_path = os.path.join(base_path, net)
    data_path = os.path.join(base_path, '3dData')

    clear_dir(net_path)

    pred_dir = os.path.join(net_path, 'pred')
    pred_files = os.listdir(pred_dir)

    if '.DS_Store' in pred_files:
        pred_files.remove('.DS_Store')

    prev_path = os.path.join(net_path, 'preview')
    ave_dc = 0

    for file in pred_files:
        pred_roi = np.load(os.path.join(pred_dir, file))
        file_num = file[0:5]

        pred_roi = process_probability_map(pred_roi)
        check_pred(pred_roi, file_num)

        img = np.load(os.path.join(data_path, 'test', file))
        roi = np.load(os.path.join(data_path, 'roi', file))

        save_pred(pred_roi, os.path.join(prev_path, 'roi_pred'), file_num)
        save_pred(img, os.path.join(prev_path, 'image'), file_num)

        rgb_true = roi_overlay(img, roi, img.shape)
        rgb_pred = roi_overlay(img, pred_roi, img.shape)

        save_pred(rgb_true, os.path.join(prev_path, 'true_overlay'), file_num)
        save_pred(rgb_pred, os.path.join(prev_path, 'pred_overlay'), file_num)

        dc = dice_coeff(roi, pred_roi)
        ave_dc += dc

        print('{} pred roi dice coefficient:\t{}'.format(file_num, dc))

    ave_dc = ave_dc / len(pred_files)

    return ave_dc


def gen_predictions(base_path, trial):
    trial_path = os.path.join(base_path, 'trial{}'.format(trial))
    data_path = os.path.join(trial_path, '3dData')

    clear_dir(data_path)

    save_from_cloud(trial)

    for net in ['2d', '3d', '3dbn', '3dvan']:
        print('{}'.format(net))
        net_dc = gen_net_prediction(base_path, net)
        print('{} overall average predictive dice coefficient:\t{}\n\n'.format(net, net_dc))

    print('Post-processing complete')


if __name__ == '__main__':
    base_path = '/Users/Matthew/Documents/Research/'
    trial = '1'
    gen_predictions(base_path, trial)
    print('fin')


