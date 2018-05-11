import os
import numpy as np

import matplotlib.pyplot as plt

import scipy.io as sio
from scipy.misc import imsave


from subprocess import call

from utilities import clear_dir

from data import cut_window
from data import normalize_image

from generator import get_generator

from post_processing import save_from_cloud

from post_processing import save_3d


def generator_test():
    save_dir = '/Users/Matthew/Documents/Research/test/preview'
    clear_dir(save_dir)

    generator = get_generator(net='cnn2d',
                              data_path='/Users/Matthew/Documents/Research/3dData/train',
                              input_shape=(32,32,48),
                              batch_size=200)

    batch = 0
    for x, y in generator:
        print(batch)
        if batch == 25:
            break

        batch += 1


def window_test():
    data_dir = '/Users/Matthew/Documents/Research/3dData'

    train_data_list = os.listdir(os.path.join(data_dir, 'train', 'image'))

    if '.DS_Store' in train_data_list:
        train_data_list.remove('.DS_Store')

    save_dir = '/Users/Matthew/Documents/Research/preview'
    clear_dir(save_dir)

    shape = (48,48,40)
    for i in train_data_list:
        name = i[0:5]
        img = np.load(os.path.join(data_dir, 'train', 'image', i))
        roi = np.load(os.path.join(data_dir, 'train', 'roi', i))

        roi = normalize_image(roi)

        img, roi, dx, dy = cut_window(shape, img, roi)

        prox_slice = roi[:, :, 0]
        dist_slice = roi[:, :, 39]

        print('{}: {} {}'.format(name, roi.shape, np.sum(roi)))

        imsave(os.path.join(save_dir, '{}prox.jpg'.format(name)), prox_slice)
        imsave(os.path.join(save_dir, '{}dist.jpg'.format(name)), dist_slice)


def graph_test_sub(subplot):
    plt.subplot(subplot)
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.plot([1, 2, 3], [1, 3, 9])

def graph_test():
    plt.figure(1)
    for i in range(1,4):
        graph_test_sub(220+i)
    plt.show()


def subprocess_test():
    cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/3dData'
    local_dir = '/Users/Matthew/Documents/Research/'
    call(['scp', '-r', cloud_dir, local_dir])


def clear_dir_test():
    dir = '/Users/Matthew/Documents/Research/preview'
    call(['find', '{}'.format(dir), '-type', 'f', '-name', '*', '-exec', 'rm', '--', '{}', '+'])

    # call(["find . ! -name '.*' ! -type d -exec rm -- {} +"])


def pred_roi_shape_test():
    base_path = '/Users/Matthew/Documents/Research/'
    trial = 1
    net_list = ['cnn2d', 'cnn3d', 'cnnbn', 'cnnwo']

    clear_dir(os.path.join(base_path, 'trial{}'.format(trial)))
    save_from_cloud(trial)

    print('Trial: {}\n'.format(trial))
    for net in net_list:
        print('Net: {}'.format(net))
        pred_dir = os.path.join(base_path, 'trial{}'.format(trial), net, 'pred')

        pred_files = os.listdir(pred_dir)
        if '.DS_Store' in pred_files:
            pred_files.remove('.DS_Store')

        test_dir = os.path.join(base_path, 'trial{}'.format(trial), '3dData', 'test', 'image')

        for file in pred_files:
            pred = np.load(os.path.join(pred_dir, file))
            img = np.load(os.path.join(test_dir, file))

            print('\t', file[0:5], ': ', img.shape, pred.shape)


def mask_test():
    mat_dir = '/Users/Matthew/Documents/Research/corrMTRdata'

    mat_files = os.listdir(mat_dir)

    if '.DS_Store' in mat_files:
        mat_files.remove('.DS_Store')

    mat_dict = sio.loadmat(os.path.join(mat_dir, mat_files[0]))

    mask = mat_dict['bgMASK']

    count(mask)

    mask = normalize_image(mask)

    preview_dir = '/Users/Matthew/Documents/Research/test/preview'
    for i in range(mask.shape[2]):
        imsave(os.path.join(preview_dir, 'maskslice{}.jpg'.format(i)), mask[:, :, i])

    count(mask)


def count(mask):
    zero_count = 0
    one_count = 0
    other_count = 0

    for it in np.nditer(mask):
        if it == 0:
            zero_count += 1
        elif it == 1:
            one_count += 1
        else:
            other_count += 1

    print('zero count: {}'.format(zero_count))
    print('one count: {}'.format(one_count))
    print('other count: {}'.format(other_count))


def rater_comparison():
    preview_dir = '/Users/Matthew/Documents/Research/test/preview'

    clear_dir(preview_dir)

    mat_dir = '/Users/Matthew/Documents/Research/corrMTRdata'
    file_1 = 'corrDATA_10011'
    file_2 = '{}_rater2'.format(file_1)

    file_1 = file_1 + '.mat'
    file_2 = file_2 + '.mat'

    mat_dict_1 = sio.loadmat(os.path.join(mat_dir, file_1))
    mat_dict_2 = sio.loadmat(os.path.join(mat_dir, file_2))

    img_1 = mat_dict_1['MTwcorr']
    img_2 = mat_dict_2['MTwcorr']
    roi_1 = mat_dict_1['nerveROI']
    roi_2 = mat_dict_2['nerveROI']

    img_1 = normalize_image(img_1)
    img_2 = normalize_image(img_2)
    roi_1 = normalize_image(roi_1)
    roi_2 = normalize_image(roi_2)

    print('IMG 1:')
    count(img_1)
    print('ROI 1:')
    count(roi_1)
    print('IMG 2:')
    count(img_2)
    print('ROI 2:')
    count(roi_2)

    for i in range(roi_1.shape[2]):
        imsave(os.path.join(preview_dir, 'rater1slice{}.jpg'.format(i)), roi_1[:, :, i])
    for i in range(roi_2.shape[2]):
        imsave(os.path.join(preview_dir, 'rater2slice{}.jpg'.format(i)), roi_2[:, :, i])


def preview_test():
    prev_dir = '/Users/Matthew/Documents/Research/test/preview'
    clear_dir(prev_dir)

    pred_dir = '/Users/Matthew/Documents/Research/trial/trial1/cnnbn/pred/'
    file_path = os.path.join(pred_dir, '10040.npy')

    pred = np.load(file_path)

    save_3d(pred, prev_dir, '10040')


def merge_folders():
    base_dir = '/Users/Matthew/Documents/Research/'
    dir1 = os.path.join(base_dir, 'biasFieldCorrData')
    dir2 = os.path.join(base_dir, 'corrMTRdata')

    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    files1.remove('.DS_Store')
    files2.remove('.DS_Store')

    overlap = set(files1).intersection(files2)
    overlap = list(overlap)

    xor = set(files1).symmetric_difference(files2)
    xor2 = xor.intersection(set(files2))

    xor2 = list(xor2)

    for i in xor2:
        print(i)

    print('merge complete')


def process_prob_map_test():
    base_path = '/Users/Matthew/Documents/Research/trial/trial1/cnnbn/pred'
    file_num = '2285500.npy'

    file_path = os.path.join(base_path, file_num)

    pred = np.load(file_path)

    countGT = 0
    countLT = 0

    meanGT = 0
    meanLT = 0

    atomicCount = 0

    for ii in np.nditer(pred):
        if ii != 1 and ii != 0:
            if ii >= .5:
                countGT += 1
                meanGT += ii
            else:
                countLT += 1
                meanLT += ii
        else:
            atomicCount += 1

    meanGT = meanGT / countGT
    meanLT = meanLT / countLT

    print('atomic count: {}'.format(atomicCount))

    print('non atomic countGT: {}'.format(countGT))
    print('non atomic countLT: {}'.format(countLT))

    print('non atomic meanGT: {}'.format(meanGT))
    print('non atomic meanLT: {}'.format(meanLT))

    for ii in np.nditer(pred, op_flags=['readwrite']):
        if ii >= .5:
            ii[...] = 1
        else:
            ii[...] = 0

    atomicCount = 0
    fail = 0

    for ii in np.nditer(pred):
        if ii == 1:
            atomicCount += 1
        elif ii == 0:
            atomicCount += 1
        else:
            fail += 1

    print('atomic count: {}'.format(atomicCount))
    print('fail: {}'.format(fail))


def roi_pixel_test():
    dir_path = '/Users/Matthew/Documents/Research/biasFieldCorrData/'
    file_path = os.path.join(dir_path, 'corrDATA_2146490.mat')

    mat_dict = sio.loadmat(file_path)
    ROI = mat_dict['nerveROI']
    ROI_slice = ROI[:, :, 0]

    zero_pix = 0
    one_pix = 0
    inval_pix = 0

    for i in np.nditer(ROI_slice):
        if i == 0:
            zero_pix += 1
        elif i == 0:
            one_pix += 1
        else:
            inval_pix += 1

    print('zero pix: ', zero_pix)
    print('one pix: ', one_pix)
    print('inval pix: ', inval_pix)
    print('ROI ave: ', np.sum(ROI)/ROI.shape[2])

    norm_ROI = normalize_image(ROI_slice)
    print('\nnormalize ROI\n')

    print('ROI ave: ', np.sum(norm_ROI))


if __name__ == '__main__':
    roi_pixel_test()
    print('testing finished')



