import numpy as np
import os

import scipy.io as sio
# from scipy.misc import imsave
from subprocess import call

from utilities import clear_dir, roi_overlay

from data import normalize_image

from matplotlib.image import imsave
import matplotlib.pyplot as plt

from post_processing import process_file
from post_processing import nerve_size

from scipy import stats
from scipy.io import loadmat


def save_from_cloud(base_path, trial_num):
    trial = str(trial_num)
    CMT_cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/CMT1AResults/'
    if trial == 'all':
        cloud_dir = CMT_cloud_dir
        local_dir = base_path
    elif trial in [str(n) for n in range(0,10)]:
        cloud_dir = os.path.join(CMT_cloud_dir, 'trial{}'.format(trial_num))
        local_dir = os.path.join(base_path,'CMT1AResults')
    else:
        raise Exception('Unknown trial parameter passed to save_from_cloud')

    print('Saving cloud data from {} to {}'.format(cloud_dir, local_dir))

    call(['scp', '-r', cloud_dir, local_dir])
    call(['scp', '-r', os.path.join(CMT_cloud_dir, '3dData'), local_dir])


def gen_predictions(mat_path, results_path, trial_path):
    data_path = os.path.join(results_path, '3dData')

    pred_dir = os.path.join(trial_path, 'pred')
    pred_files = os.listdir(pred_dir)

    if '.DS_Store' in pred_files:
        pred_files.remove('.DS_Store')

    prev_path = os.path.join(trial_path, 'preview')

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

        if dc_dict[file_num] >= .6:
            # pred_dict[file_num] = get_pred_nerve_size(pred_roi)
            pred_roi = normalize_image(pred_roi)
            pred_dict[file_num] = np.sum(pred_roi) / pred_roi.shape[2]

    return dc_dict, pred_dict


def gen_trial_predictions(results_path, mat_dir, trial):
    print('Trial {} Predictions: '.format(trial))
    trial_path = os.path.join(results_path, 'trial{}'.format(trial))

    dc_dict, pred_dict = gen_predictions(mat_path=os.path.join(base_path, mat_dir),
                                             results_path=results_path,
                                             trial_path=trial_path)

    successful_ave_dc = dc_dict['successful_ave_dc'] / (dc_dict['num_file'] - dc_dict['num_fail'])
    overall_ave_dc = dc_dict['overall_ave_dc'] / dc_dict['num_file']
    cmt_patient_ave_dc = dc_dict['cmt_patient_ave_dc'] / dc_dict['num_cmt_patient']

    print('\tCMT1A successful predictive dice coefficient:\t{}'.format(successful_ave_dc))
    print('\tCMT1A average overall predictive dice coefficient:\t{}'.format(overall_ave_dc))
    print('\tCMT1A average CMT predictive dice coefficient:\t\t{}'.format(cmt_patient_ave_dc))

    # if net == 'cnnbn':
    #     print(cmt_pred_dict)

    CMT1A_graphing(pred_dict=pred_dict,
                   mat_dir=mat_dir)

    print('Trial {} Post-processing complete\n'.format(trial))


def get_nerve_size(file_num, mat_dir):
    mat_files = os.listdir(mat_dir)

    if '.DS_Store' in mat_files:
        mat_files.remove('.DS_Store')

    for mat_file in mat_files:
        if '{}.mat'.format(file_num) in mat_file:
            mat_dict = loadmat(os.path.join(mat_dir, mat_file))
            roi = mat_dict['nerveROI']
            avg_nerve_area = 0

            # for i in range(roi.shape[2]):
            #     avg_nerve_area = avg_nerve_area + np.sum(roi[:, :, i])
            #
            # avg_nerve_area = avg_nerve_area / roi.shape[2]

            roi = normalize_image(roi)
            avg_CSA = np.sum(roi) / roi.shape[2]

            return avg_CSA

    print('{} associated .mat file was not found'.format(file_num))


# def get_pred_nerve_size(pred_roi):
#     avg_nerve_area = 0
#     for i in range(pred_roi.shape[2]):
#         avg_nerve_area = avg_nerve_area + np.sum(pred_roi[:, :, i])
#
#     avg_nerve_area = avg_nerve_area / pred_roi.shape[2]
#
#     return avg_nerve_area


def CMT1A_graphing(pred_dict, mat_dir):
    CMTES = {}
    CMTES[20010]=12
    CMTES[20020]=10
    CMTES[20022]='NA'
    CMTES[20030]=1
    CMTES[20070]='NA'
    CMTES[20090]=13
    CMTES[20150]=17
    CMTES[20160]=17
    CMTES[20180]=3
    CMTES[20200]=5
    CMTES[20210]=9
    CMTES[20310]=17
    CMTES[1150560]=5
    CMTES[1150570]=14
    CMTES[1162010]=13
    CMTES[1279360]=12
    CMTES[1279830]=6
    CMTES[2131220]=17
    CMTES[2146490]=18
    CMTES[2161250]=17

    for file_num in pred_dict.copy():
        file_num = int(file_num)

        if CMTES[file_num] != 'NA':
            pred_dict['{}'.format(file_num)] = [pred_dict['{}'.format(file_num)], CMTES[file_num]]
        else:
            del pred_dict['{}'.format(file_num)]

    file_list = []
    pred_nerve_size_list = []
    nerve_size_list = []
    CMTES_list = []

    # python 3
    for file_num in pred_dict.copy():
        nerve_size_list.append(get_nerve_size(file_num=file_num,
                                              mat_dir=mat_dir))
        pred_nerve_size_list.append((pred_dict[file_num])[0])
        CMTES_list.append(pred_dict[file_num][1])
        file_list.append(file_num)

    nerve_lists = [pred_nerve_size_list, nerve_size_list]

    for i in range(len(file_list)):
        print('File: ', file_list[i])
        print('\tCMTES: ', CMTES_list[i])
        print('\tHand CSA: ', nerve_size_list[i])
        print('\tCNN CSA: ', pred_nerve_size_list[i], '\n')

    for list in nerve_lists:
        slope, intercept, r_value, p_value, std_err = stats.linregress(CMTES_list, list)

        print('slope:\t{}'.format(slope))
        print('intercept:\t', intercept)
        print('beta:\t{}'.format(r_value))
        print('std_error:\t{}'.format(std_err))

        best_fit = [intercept + slope * i for i in CMTES_list]
        print('Best Fit: ', best_fit)

        plt.plot(CMTES_list, list, 'o', label='original data')
        plt.plot(CMTES_list, best_fit, 'r--', label='fitted line')
        plt.ylabel('Cross Sectional Area')
        plt.xlabel('CMTES')
        plt.legend()
        plt.show()

        print('\n')


if __name__ == '__main__':
    base_path = '/Users/Matthew/Documents/Research/'

    # if(os.path.isdir(os.path.join(base_path, 'CMT1AResults'))):
    #     clear_dir(os.path.join(base_path, 'CMT1AResults'))
    #
    #
    # save_from_cloud(base_path, 'all')


    # for i in range(1,4):
    for i in range(8,9):
        trial = i
        gen_trial_predictions(results_path=os.path.join(base_path, 'CMT1AResults'),
                              mat_dir=os.path.join(base_path, 'biasFieldCorrData'),
                              trial=trial)

    print('fin')


