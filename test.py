import os
import numpy as np

import matplotlib.pyplot as plt

from scipy.misc import imsave

from subprocess import call

from clear_directories import clear_dir

from data import cut_window
from data import normalize_image

from generator import get_generator


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


def graph_test():
    plt.figure(1)
    plt.subplot(211)
    plt.plot([1,2,3])
    plt.subplot(211)
    plt.plot([4,5,6], 'r--')
    plt.show()


def subprocess_test():
    cloud_dir = 'hancocmp@login.accre.vanderbilt.edu:/scratch/hancocmp/3dData'
    local_dir = '/Users/Matthew/Documents/Research/'
    call(['scp', '-r', cloud_dir, local_dir])


def clear_dir_test():
    dir = '/Users/Matthew/Documents/Research/preview'
    call(['find', '{}'.format(dir), '-type', 'f', '-name', '*', '-exec', 'rm', '--', '{}', '+'])

    # call(["find . ! -name '.*' ! -type d -exec rm -- {} +"])

if __name__ == '__main__':
    window_test()
    print('testing finished')



