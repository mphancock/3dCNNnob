import numpy as np
import os

import scipy.io as sio
# from scipy.misc import imsave
from subprocess import call

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
    non_zero = False
    count = 0
    for i in np.nditer(pred_roi):
        if i != 1 and i != 0:
            non_zero = True
            count += 1

    if non_zero:
        print('\t\t{} has {} non atomic element out of {} elements'.format(num, count, pred_roi.size))


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


def save_from_cloud(base_path, trial_num):
    trial = str(trial_num)
    trial_cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/trial/'
    if trial == 'all':
        cloud_dir = trial_cloud_dir
    elif trial in [str(n) for n in range(0,10)]:
        cloud_dir = os.path.join(trial_cloud_dir, 'trial{}'.format(trial_num))
    else:
        raise Exception('Unknown trial parameter passed to save_from_cloud')

    print('Saving cloud data from: {}'.format(cloud_dir))
    local_dir = os.path.join(base_path, 'trial')
    call(['scp', '-r', cloud_dir, local_dir])


def mask_processing(img, num, mat_path):
    mat_files = os.listdir(mat_path)

    if '.DS_Store' in mat_files:
        mat_files.remove('.DS_Store')

    count = 0

    file = ''
    for i in mat_files:
        if '.mat' in i:
            count += 1

            if num in i:
                file = i
                break
            count += 1

    # print(file)
    # print(num)
    # print(count)

    if file == '':
        raise Exception('Could not find corresponding mat file to: '.format(num))

    # print('File: {}'.format(file))

    mat_dict = sio.loadmat(os.path.join(mat_path, file))
    mask = mat_dict['bgMASK']

    mask = normalize_image(mask)
    if(mask.shape != (256, 256, 40)):
        print('\t\t{} incorrect mask shape: {}'.format(num, mask.shape))
        return img

    mask_img = np.multiply(img, mask)

    return mask_img


def analyze_dc(dc, file_num, dc_dict):
    print('\t\t{} pred roi dice coefficient:\t{}'.format(file_num, dc))

    dc_dict[file_num] = dc

    if dc > .5:
        dc_dict['successful_ave_dc'] += dc
    else:
        dc_dict['num_fail'] += 1
        print('\t\t\tSegmentation failed')

    dc_dict['overall_ave_dc'] += dc

    if file_num[0] == '2':
        dc_dict['num_cmt_patient'] += 1
        dc_dict['cmt_patient_ave_dc'] += dc

    return dc_dict


def process_probability_map(pred):
    for it in np.nditer(pred, op_flags=['readwrite']):
        if it >= .5:
            it[...] = 1
        else:
            it[...] = 0

    return pred


def nerve_size(pred):
    medial_slice = pred[:,:,20]

    return np.sum(medial_slice)


def process_file(raw_pred_roi, file_num, mat_path, data_path, prev_path, dc_dict):
    # check_pred(raw_pred_roi, file_num)

    adjusted_pred_roi = process_probability_map(raw_pred_roi)

    # check_pred(adjusted_pred_roi, file_num)

    # print('File num: {}'.format(file_num))
    pred_roi = mask_processing(img=adjusted_pred_roi,
                               num=file_num,
                               mat_path=mat_path)

    img = np.load(os.path.join(data_path, 'test', 'image', file_num+'.npy'))
    roi = np.load(os.path.join(data_path, 'test', 'roi', file_num+'.npy'))

    save_3d(pred_roi, os.path.join(prev_path, 'roi_pred'), file_num)
    save_3d(img, os.path.join(prev_path, 'image'), file_num)

    rgb_true = roi_overlay(img, roi)
    rgb_pred = roi_overlay(img, pred_roi)

    rgb_true = rgb_true.transpose(1, 2, 0, 3)
    rgb_pred = rgb_pred.transpose(1, 2, 0, 3)

    save_3d(rgb_true, os.path.join(prev_path, 'true_overlay'), file_num)
    save_3d(rgb_pred, os.path.join(prev_path, 'pred_overlay'), file_num)

    dc = dice_coeff(roi, pred_roi)

    analyze_dc(dc=dc,
               file_num=file_num,
               dc_dict=dc_dict)

    return pred_roi, dc_dict

def gen_net_prediction(mat_path, trial_path, net):
    net_path = os.path.join(trial_path, net)
    data_path = os.path.join(trial_path, '3dData')

    pred_dir = os.path.join(trial_path, net, 'pred')
    pred_files = os.listdir(pred_dir)

    if '.DS_Store' in pred_files:
        pred_files.remove('.DS_Store')

    prev_path = os.path.join(trial_path, net, 'preview')

    dc_dict = dict()

    dc_dict['successful_ave_dc'] = 0
    dc_dict['overall_ave_dc'] = 0
    dc_dict['cmt_patient_ave_dc'] = 0
    dc_dict['num_fail'] = 0
    dc_dict['num_cmt_patient'] = 0
    dc_dict['num_file'] = len(pred_files)

    pred_dict = {}

    for file in pred_files:
        pred_roi = np.load(os.path.join(pred_dir, file))

        if file[5] in [str(n) for n in range(0, 10)]:
            file_num = file[0:7]
        else:
            file_num = file[0:5]

        pred_roi, dc_dict = process_file(raw_pred_roi=pred_roi,
                                         file_num=file_num,
                                         mat_path=mat_path,
                                         data_path=data_path,
                                         prev_path=prev_path,
                                         dc_dict=dc_dict)

        if dc_dict[file_num] >= .5:
            pred_dict[file_num] = nerve_size(pred_roi)

    return dc_dict, pred_dict


def gen_trial_predictions(base_path, mat_dir, trial):
    print('Trial {} Predictions: '.format(trial))
    trial_path = os.path.join(base_path, 'trial', 'trial{}'.format(trial))

    for net in ['cnn2d', 'cnn3d', 'cnnbn', 'cnnwo']:
        print('\tNet: {}'.format(net))
        dc_dict, cmt_pred_dict = gen_net_prediction(mat_path=os.path.join(base_path, mat_dir),
                                     trial_path=trial_path,
                                     net=net)

        successful_ave_dc = dc_dict['successful_ave_dc'] / (dc_dict['num_file'] - dc_dict['num_fail'])
        overall_ave_dc = dc_dict['overall_ave_dc'] / dc_dict['num_file']
        cmt_patient_ave_dc = dc_dict['cmt_patient_ave_dc'] / dc_dict['num_cmt_patient']

        print('\t{} average successful predictive dice coefficient:\t{}'.format(net, successful_ave_dc))
        print('\t{} average overall predictive dice coefficient:\t{}'.format(net, overall_ave_dc))
        print('\t{} average CMT predictive dice coefficient:\t\t{}'.format(net, cmt_patient_ave_dc))

        if net == 'cnnbn':
            print(cmt_pred_dict)

    print('Trial {} Post-processing complete\n'.format(trial))


if __name__ == '__main__':
    base_path = '/Users/Matthew/Documents/Research/'

    # if(os.path.isdir(os.path.join(base_path, 'trial'))):
    #     clear_dir(os.path.join(base_path, 'trial'))
    #
    # save_from_cloud(base_path, 1)

    # for i in range(1,4):
    for i in range(1,2):
        trial = i
        gen_trial_predictions(base_path=base_path,
                              mat_dir='biasFieldCorrData',
                              trial=trial)

    print('fin')


