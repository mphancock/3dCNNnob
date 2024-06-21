import os
import numpy as np

from sklearn.metrics import normalized_mutual_info_score as nmi

from scipy.ndimage.measurements import center_of_mass as com

from subprocess import call

def get_count(path):
    files = os.listdir(path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    return len(files)


def clear_dir(dir):
    call(['find', '{}'.format(dir), '-type', 'f', '-name', '*', '-exec', 'rm', '--', '{}', '+'])
    print('{} is clear'.format(dir))


def affine_registration(shape, img_stack, roi_stack, img):
    max_nmi = 0
    max_nmi_index = 0

    for i in range(img_stack.shape[0]):
        temp_img = img_stack[i, :, :, :]

        nmi_score = nmi(temp_img.reshape(-1), img.reshape(-1))

        if nmi_score > max_nmi:
            max_nmi = nmi_score
            max_nmi_index = i

    roi_base = roi_stack[max_nmi_index, :, :, :]

    x_com, y_com, z_com = com(roi_base)

    rx_com = int(round(x_com))
    ry_com = int(round(y_com))

    dx = rx_com - int(shape[0]/2)
    dy = ry_com - int(shape[0]/2)

    if dx < 0:
        dx = 0

    if dy < 0:
        dy = 0

    dx = int(dx)
    dy = int(dy)

    c_img = img[dx:dx+shape[0], dy:dy+shape[1], :]

    return c_img, dx, dy


def roi_overlay(img, roi):
    rgb = np.ndarray([img.shape[2], img.shape[0], img.shape[1], 3])

    img_t = img.T
    roi_t = roi.T

    rgb[:, :, :, 0] = img_t

    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for k in range(rgb.shape[2]):
                if roi_t[i, j, k] > .5:
                    rgb[i, j, k, 0] = 1

    rgb[:, :, :, 1] = img_t
    rgb[:, :, :, 2] = img_t

    return rgb











