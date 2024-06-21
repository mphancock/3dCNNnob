import os

import numpy as np

from data import zero_pad
from data import cut_window

from data import cut_window_2d

from augment import elastic_transform
from augment import elastic_transform_2d


def add_data_3d(img, roi, image_list, roi_list, input_shape, augment):
    window_shape = (input_shape[0], input_shape[1], 40)

    img, roi, dx, dy = cut_window(window_shape, img, roi)

    img = zero_pad(img, input_shape)
    roi = zero_pad(roi, input_shape)

    if augment:
        img, roi = elastic_transform(img, roi)

    img = np.reshape(img, (1,) + img.shape)
    roi = np.reshape(roi, (1,) + roi.shape)

    image_list.append(img)
    roi_list.append(roi)

    # print('\tvolume: {}'.format(np.sum(np.squeeze(img))))


def add_data_2d(img, roi, image_list, roi_list, input_shape, augment):
    img, roi, dx, dy = cut_window(input_shape, img, roi)

    for i in range(img.shape[2]):
        img_slice = img[:, :, i]
        roi_slice = roi[:, :, i]

        if augment:
            img_slice, roi_slice = elastic_transform_2d(img_slice, roi_slice)

        img_slice = np.reshape(img_slice, (1,) + img_slice.shape)
        roi_slice = np.reshape(roi_slice, (1,) + roi_slice.shape)

        image_list.append(img_slice)
        roi_list.append(roi_slice)

        # if i == 0:
        #     sys.stdout.write('\tproximal: ')
        #     sys.stdout.write('{}'.format((np.sum(np.squeeze(img_slice))).astype(int)))
        # if i == 39:
        #     print('\tdistal:', (np.sum(np.squeeze(img_slice))).astype(int))


def generator(data_type, data_path, input_shape, batch_size, augment):
    data_list = os.listdir(os.path.join(data_path, 'image'))

    if '.DS_Store' in data_list:
        data_list.remove('.DS_Store')

    while True:
        image_list = list()
        roi_list = list()

        for file in data_list:
            img = np.load(os.path.join(data_path, 'image', file))
            roi = np.load(os.path.join(data_path, 'roi', file))

            if data_type == '3d':
                add_data_3d(img, roi, image_list, roi_list, input_shape, augment)
            elif data_type == '2d':
                add_data_2d(img, roi, image_list, roi_list, input_shape, augment)

            if len(image_list) == batch_size:
                x, y = np.asarray(image_list), np.asarray(roi_list)
                yield x, y

                image_list = list()
                roi_list = list()


def get_generator(net, data_path, input_shape, batch_size):
    if net == 'cnn3d' or net == 'cnnbn':
        return generator(data_type='3d',
                         data_path=data_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         augment=True)
    elif net == 'cnnwo':
        return generator(data_type='3d',
                         data_path=data_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         augment=False)
    elif net == 'cnn2d':
        return generator(data_type='2d',
                         data_path=data_path,
                         input_shape=input_shape,
                         batch_size=batch_size,
                         augment=True)
    else:
        raise RuntimeError('Invalid net type passed to get_generator')




