import numpy as np
import os
import math

from scipy.misc import imsave
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy.ndimage.measurements import center_of_mass as com\

from utilities import get_count

split_pct = .8


def cut_window(shape, img, roi):
    x_com, y_com, z_com = com(roi)

    x_com = round(x_com)
    y_com = round(y_com)

    dx = int(x_com - int(shape[0]/2))
    dy = int(y_com - int(shape[1]/2))

    if dx < 0:
        dx = 0

    if dy < 0:
        dy = 0

    c_img = img[dx:dx+shape[0], dy:dy+shape[1], :]
    c_roi = roi[dx:dx+shape[0], dy:dy+shape[1], :]

    return c_img, c_roi, dx, dy


def zero_pad(image, shape):

    delta1 = int(math.fabs(image.shape[0] - shape[0]))
    delta2 = int(math.fabs(image.shape[1] - shape[1]))
    delta3 = int(math.fabs(image.shape[2] - shape[2]))

    delta1r = int(delta1/2)
    delta1l = delta1 - delta1r

    delta2r = int(delta2/2)
    delta2l = delta2 - delta2r

    delta3r = int(delta3/2)
    delta3l = delta3 - delta3r

    image = np.lib.pad(image, ((delta1l, delta1r), (delta2l, delta2r), (delta3l, delta3r)), 'constant', constant_values=0)

    return image


def normalize_image(input_img):
    maxval = np.amax(input_img)
    outimg = input_img/maxval

    return outimg


def get_stack(data_path, img_shape):
    num_val = get_count(os.path.join(data_path, 'val', 'image'))
    num_train = get_count(os.path.join(data_path, 'test', 'image'))

    img_stack = np.ndarray([num_train + num_val, img_shape[0], img_shape[1], img_shape[2]])
    roi_stack = np.ndarray([num_train + num_val, img_shape[0], img_shape[1], img_shape[2]])

    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    train_file_list = os.listdir(os.path.join(train_dir, 'image'))
    val_file_list = os.listdir(os.path.join(val_dir, 'image'))

    for i in range(num_train + num_val):
        if i < num_train:
            img_stack[i, :, :, :] = np.load(os.path.join(train_dir, 'image', train_file_list[i]))
            roi_stack[i, :, :, :] = np.load(os.path.join(train_dir, 'roi', train_file_list[i]))
        else:
            x = i - num_train
            img_stack[i, :, :, :] = np.load(os.path.join(val_dir, 'image', val_file_list[x]))
            roi_stack[i, :, :, :] = np.load(os.path.join(val_dir, 'roi', val_file_list[x]))

    return img_stack, roi_stack


def resize_roi(pred_roi, dx, dy, img_shape):
    x_pad = (dx, img_shape[0]-pred_roi.shape[0]-dx)
    y_pad = (dy, img_shape[1]-pred_roi.shape[1]-dy)

    resized_roi = np.lib.pad(pred_roi, (x_pad, y_pad, (0,0)), 'constant', constant_values=0)

    return resized_roi


def save_image(image, directory, name):
    file_extension = '.jpg'
    for i in range(image.shape[2]):
        imsave(os.path.join(directory, 'image', '{}image{}{}'.format(name, i, file_extension)), image[:, :, i])


def cut_window_2d(shape, img, roi):
    x_com, y_com = com(roi)

    x_com = round(x_com)
    y_com = round(y_com)

    dx = int(x_com - int(shape[0]/2))
    dy = int(y_com - int(shape[1]/2))

    if dx < 0:
        dx = 0

    if dy < 0:
        dy = 0

    c_img = img[dx:dx+shape[0], dy:dy+shape[1]]
    c_roi = roi[dx:dx+shape[0], dy:dy+shape[1]]

    return c_img, c_roi
