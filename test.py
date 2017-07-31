import os
import numpy as np
import scipy.io as sio

from scipy.misc import imsave
from matplotlib.image import imread

from data import normalize_image
from data import zero_pad
from data import save_image
from data import affine_registration
from data import get_stack

from generator import get_generator

from data import resize_roi

from save_data import save_data

from copy import deepcopy

from utilities import sum_border

from random import shuffle

from clear_directories import clear_data_directories


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




if __name__ == '__main__':
    affine_test()
    print('testing complete')





