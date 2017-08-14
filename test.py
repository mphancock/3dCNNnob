import os
import numpy as np
import scipy.io as sio
import sys
import matplotlib.pyplot as plt

from scipy.misc import imsave
from matplotlib.image import imread

from subprocess import call

from data import normalize_image
from data import zero_pad
from data import save_image
from data import affine_registration
from data import get_stack
from data import cut_window
from data import cut_window2d

from generator import get_generator

from data import resize_roi

from augment import elastic_transform

from save_data import save_data

from copy import deepcopy

from utilities import sum_border

from random import shuffle

from clear_directories import clear_data_directories

from utilities import roi_overlay

from post import get_data



def generator_test():
    test_file = '/Users/Matthew/Documents/Research/test/test3dData'
    save_file = '/Users/Matthew/Documents/Research/test/preview'

    generator = get_generator(test_file, batch_size=1)

    batch = 0
    count = 0

    for x, y in generator:
        print(count)
        print(batch)
        if count > 3:
            break

        print(x.shape)
        image = x.squeeze()
        roi = y.squeeze()

        if batch%4 == 0:
            save_image(image, roi, save_file, 'batch{}'.format(batch))
            count += 1

        batch += 1


def resize_roi_test():
    pred_roi = np.ndarray([3,3,4])

    for i in range(pred_roi.shape[2]):
        array = np.ndarray([pred_roi.shape[0], pred_roi.shape[1]])
        array.fill(i)
        pred_roi[:, :, i] = array

    img_shape = (10,10,4)

    roi = resize_roi(pred_roi, dx=1, dy=1, img_shape=img_shape)
    print(roi)
    print(roi[1:4, 1:4, :])

    for i in range(roi.shape[2]):
        print('slice{}'.format(i))
        print(roi[:, :, i])


def resize_test():
    x = np.ndarray([220, 220, 40])
    y = np.ndarray([220, 220, 40])

    x, y = zero_pad(x,y)
    print('image shape: {}'.format(x.shape))
    print('roi shape: {}'.format(y.shape))

    x = np.ndarray([219, 219, 39])
    y = np.ndarray([219, 219, 39])

    zero_pad(x,y)

    x = np.ndarray([260, 260, 41])
    y = np.ndarray([260, 260, 41])

    zero_pad(x,y)


def data_test():
    mat_path = '/Users/Matthew/Documents/Research/CorrectedMTRData'
    data_path = '/Users/Matthew/Documents/Research/3dData'

    save_data(mat_path, data_path)


def affine_test():
    base_path = '/Users/Matthew/Documents/Research/3dData'

    clear_data_directories(base_path)

    mat_path = '/Users/Matthew/Documents/Research/corrMTRdata'
    save_data(mat_path, base_path)

    test_files = os.listdir(os.path.join(base_path, 'test', 'image'))

    shuffle(test_files) 

    file = test_files[0]
    print(file)

    img_shape = 256, 256, 40
    new_shape = 48, 48, 40

    data_path = '/Users/Matthew/Documents/Research/3dData'
    preview_dir = '/Users/Matthew/Documents/Research/test/preview'

    img_stack, roi_stack = get_stack(data_path, img_shape)

    img = np.load(os.path.join(data_path, 'test', 'image', file))
    roi = np.load(os.path.join(data_path, 'test', 'roi', file))

    c_img, dx, dy = affine_registration(new_shape, img_stack, roi_stack, img)

    print(c_img.shape)

    print(dx)
    print(dy)

    roi_window = roi[dx:dx+new_shape[0], dy:dy+new_shape[1], :]
    print('roi sum: {}'.format(np.sum(roi_window)))
    print('roi border sum: {}'.format(sum_border(roi_window)))

    for j in range(c_img.shape[2]):
        img_name = 'img{}.jpg'.format(j)
        roi_name = 'roi{}.jpg'.format(j)

        imsave(os.path.join(preview_dir, img_name), c_img[:, :, j])
        imsave(os.path.join(preview_dir, roi_name), roi_window[:, :, j])


def affine_test_2():
    file = '11643.npy'
    print(file)

    img_shape = 256, 256, 40
    new_shape = 48, 48, 40

    data_path = '/Users/Matthew/Documents/Research/3dData'
    preview_dir = '/Users/Matthew/Documents/Research/test/preview'

    img_stack, roi_stack = get_stack(data_path, img_shape)

    img = np.load(os.path.join(data_path, 'test', 'image', file))
    roi = np.load(os.path.join(data_path, 'test', 'roi', file))

    c_img, dx, dy = affine_registration(new_shape, img_stack, roi_stack, img)

    print(c_img.shape)

    print(dx)
    print(dy)

    roi_window = roi[dx:dx+new_shape[0], dy:dy+new_shape[1], :]
    print('roi sum: {}'.format(np.sum(roi_window)))

    for j in range(c_img.shape[2]):
        img_name = 'img{}.jpg'.format(j)
        roi_name = 'roi{}.jpg'.format(j)

        imsave(os.path.join(preview_dir, img_name), c_img[:, :, j])
        imsave(os.path.join(preview_dir, roi_name), roi_window[:, :, j])

    border_sum = sum_border(roi_window)

    print('border sum: {}'.format(border_sum))


def data_test2():
    dir = '/Users/Matthew/Documents/Research/corrMTRData'

    files = os.listdir(dir)

    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for i in files:
        mat_dict = sio.loadmat(os.path.join(dir, i))
        img = mat_dict['MTwcorr']
        print('name: {}\tshape:{}'.format(i[9:14], img.shape))


def save_preview():
    path = '/Users/Matthew/Downloads/aff'
    save_path = '/Users/Matthew/Documents/Research/test/preview'
    files = os.listdir(path)

    if '.DS_Store' in files:
        files.remove('.DS_Store')

    print(files)

    path = os.path.join(path, '20070.npy')

    aff = np.load(path)
    for i in range(aff.shape[2]):
        name = 'aff{}.jpg'.format(i)
        name = os.path.join(save_path, name)
        imsave(name, aff[:, :, i])


    # for i in files:
    #     file_path = os.path.join(path, i)
    #     pred = np.load(file_path)
    #     for j in range(pred.shape[2]):
    #         imsave(os.path.join(save_path, 'aff{}slice{}.jpg'.format(i, j)), pred[:, :, j])


def elastic_save():
    path = '/Users/Matthew/Documents/Research/3dData/test/image/10021.npy'
    roi_path = '/Users/Matthew/Documents/Research/3dData/test/roi/10021.npy'

    img = np.load(path)
    roi = np.load(roi_path)

    rgb = roi_overlay(img, roi, img.shape)
    save_path = '/Users/Matthew/Documents/Research/test/preview'

    for i in range(img.shape[2]):
        name = 'overlay{}.jpg'.format(i)
        name = os.path.join(save_path, name)

        imsave(name, rgb[i, :, :, :])

    img, roi = elastic_transform(img, roi)
    rgb_elastic = roi_overlay(img, roi, img.shape)

    for i in range(img.shape[2]):
        name = 'elastic_overlay{}.jpg'.format(i)
        name = os.path.join(save_path, name)

        imsave(name, rgb_elastic[i, :, :, :])


def window_save():
    path = '/Users/Matthew/Documents/Research/3dData/test/image/10021.npy'
    roi_path = '/Users/Matthew/Documents/Research/3dData/test/roi/10021.npy'

    img = np.load(path)
    roi = np.load(roi_path)

    img, roi = elastic_transform(img, roi)

    img, roi = cut_window((48,48,40), img, roi)

    rgb = roi_overlay(img, roi, img.shape)

    save_path = '/Users/Matthew/Documents/Research/test/preview'
    for i in range(img.shape[2]):
        name = 'rgb{}.jpg'.format(i)
        roi_name = 'roi{}.jpg'.format(i)

        name = os.path.join(save_path, name)
        roi_name = os.path.join(save_path, roi_name)

        imsave(name, rgb[i, :, :, :])
        imsave(roi_name, roi[:, :, i])


def analyze_img():
    np.set_printoptions(linewidth=250, precision=2)

    name = '10011.npy'

    data_path = '/Users/Matthew/Documents/Research/3dData'
    img, roi = get_data(data_path, name)

    shape = (16,16)

    slice = 0
    img_slice = img[:, :, slice]
    roi_slice = roi[:, :, slice]

    save_dir = '/Users/Matthew/Documents/Research/test/preview'
    img_save_path = os.path.join(save_dir, 'orgimgslice{}.jpg'.format(slice))
    roi_save_path = os.path.join(save_dir, 'orgroislice{}.jpg'.format(slice))

    pred_dir = '/Users/Matthew/Downloads/pred'

    pred_path = os.path.join(pred_dir, name)
    pred_roi = np.load(pred_path)

    pred_roi_slice = pred_roi[:, :, slice]
    print(pred_roi_slice.shape)
    pred_roi_save_path = os.path.join(save_dir, 'predroislice{}.jpg'.format(slice))
    imsave(pred_roi_save_path, pred_roi_slice)

    imsave(img_save_path, img_slice)
    imsave(roi_save_path, roi_slice)

    dx, dy, img_slice, roi_slice = cut_window2d(shape, img_slice, roi_slice)

    # for i in range(img.shape[2]):

    new_arr = np.ndarray(img_slice.shape)
    new_arr.fill(0)

    for i in range(img_slice.shape[0]):
        for j in range(img_slice.shape[1]):
            if roi_slice[i,j] == 1:
                new_arr[i,j] = img_slice[i,j]

    new_arr_2 = np.ndarray(img_slice.shape)
    new_arr_2.fill(0)

    new_arr_3 = np.ndarray(img_slice.shape)
    new_arr_3.fill(0)

    for i in range(img_slice.shape[0]):
        for j in range(img_slice.shape[1]):
            if roi_slice[i,j] == 1:
                new_arr_2[i,j] = img_slice[i,j]
            elif 1 < j < roi_slice.shape[1]-1 and roi_slice[i,j-1] == 1 and roi_slice[i,j+1] == 1:
                new_arr_2[i,j] = img_slice[i,j]

            else:
                new_arr_3[i,j] = img_slice[i,j]

    new_arr_4 = np.ndarray(img_slice.shape)
    new_arr_4.fill(0)

    new_arr_5 = np.ndarray(img_slice.shape)
    new_arr_5.fill(0)

    pred_roi_slice = pred_roi_slice[dx:dx+shape[0], dy:dy+shape[1]]
    for i in range(img_slice.shape[0]):
        for j in range(img_slice.shape[1]):
            if pred_roi_slice[i,j] == 1:
                new_arr_4[i,j] = img_slice[i,j]
            else:
                new_arr_5[i,j] = img_slice[i,j]



    print(np.array2string(new_arr))
    print('')
    # print(np.array2string(new_arr_2))
    # print('')
    print(np.array2string(new_arr_4))
    print('')

    print(np.array2string(new_arr_3))
    print('')
    print(np.array2string(new_arr_5))
    print('')

    # print(np.array2string(img_slice))

    img_save_path = os.path.join(save_dir, 'cutimgslice{}.jpg'.format(slice))
    roi_save_path = os.path.join(save_dir, 'cutroislice{}.jpg'.format(slice))

    imsave(img_save_path, img_slice)
    imsave(roi_save_path, roi_slice)


def graph_test():
    plt.figure(1)
    plt.subplot(211)
    plt.plot([1,2,3])
    plt.subplot(211)
    plt.plot([4,5,6], 'r--')
    plt.show()


def aff_test():
    dir = '/Users/Matthew/Downloads/aff'
    files = os.listdir(dir)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for i in files:
        file_path = os.path.join(dir, i)
        aff = np.load(file_path)
        print('{} shape: {}'.format(i, aff.shape))


def subprocess_test():
    cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/3dData'
    local_dir = '/Users/Matthew/Documents/Research/'
    call(['scp', '-r', cloud_dir, local_dir])


if __name__ == '__main__':
    subprocess_test()




