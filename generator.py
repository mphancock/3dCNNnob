import os

import numpy as np

from data import zero_pad
from data import cut_window

from augment import elastic_transform


def get_generator(data_file, batch_size, input_shape):
    return data_generator(data_file, batch_size, input_shape)


def data_generator(data_directory, batch_size, input_shape):
    file_list = os.listdir(os.path.join(data_directory, 'image'))

    if '.DS_Store' in file_list:
        file_list.remove('.DS_Store')

    while True:
        image_list = list()
        roi_list = list()
        # shuffle(file_list)

        for file in file_list:
            # print(file)
            add_data(image_list, roi_list, data_directory, file, input_shape)
            if len(image_list) == batch_size:
                x, y = np.asarray(image_list), np.asarray(roi_list)
                yield x, y

                image_list = list()
                roi_list = list()


def add_data(image_list, roi_list, data_directory, file, input_shape):
    image = np.load(os.path.join(data_directory, 'image', file))
    roi = np.load(os.path.join(data_directory, 'roi', file))

    window_shape = (input_shape[0], input_shape[0], 40)

    image, roi = cut_window(window_shape, image, roi)

    image = zero_pad(image, input_shape)
    roi = zero_pad(roi, input_shape)

    image, roi = elastic_transform(image, roi)

    image = np.reshape(image, (1,) + image.shape)
    roi = np.reshape(roi, (1,) + roi.shape)

    image_list.append(image)
    roi_list.append(roi)



