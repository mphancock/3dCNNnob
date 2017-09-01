from generator import get_generator

from utilities import get_count

from utilities import affine_registration

from data import get_stack
from data import cut_window
from data import zero_pad
from data import resize_roi

import os
import math

import numpy as np


#currently has the wrong input size for model
def evaluate_model(model):
    train_data_path = '/scratch/hancocmp/3dData/train/'
    batch_size = 3

    eval_generator = get_generator(train_data_path, batch_size=batch_size)
    num_train_files = get_count(os.path.join(train_data_path, 'image'))

    #python 2.x int / int returns int
    eval_steps = math.ceil(num_train_files / float(batch_size))

    scores = model.predict_generator(eval_generator,
                            steps=eval_steps)

    return scores


def process_probability_map(pred_roi):
    if np.mean(pred_roi) > 1e-05:
        pred_roi = pred_roi > np.mean(pred_roi)

    if np.mean(pred_roi) < 1e-05:
        pred_roi = pred_roi > 0

    return pred_roi


def predict_3d(pred_model, img, roi, input_shape):
    img_cut, roi_cut, dx, dy = cut_window(input_shape, img, roi)

    input_img_cut = zero_pad(img_cut, shape=input_shape)

    input_multi_dim = np.reshape(input_img_cut, (1,) + (1,) + input_img_cut.shape)

    output_multi_dim = pred_model.predict(input_multi_dim)

    output_pred_roi_cut = np.squeeze(output_multi_dim)

    pred_roi_cut = output_pred_roi_cut[:, :, 4:44]

    processed_roi_cut = process_probability_map(pred_roi_cut)

    return resize_roi(processed_roi_cut, dx, dy, img.shape)


def predict_2d(pred_model, img, roi, input_shape):
    pred_roi_cut = np.ndarray(input_shape + (40,))
    img_cut, roi_cut, dx, dy = cut_window(input_shape, img, roi)

    for i in range(img_cut.shape[2]):
        input_img_cut_slice = img_cut[:, :, i]

        input_multi_dim = np.reshape(input_img_cut_slice, (1,) + (1,) + input_img_cut_slice.shape)

        output_multi_dim = pred_model.predict(input_multi_dim)

        output_pred_roi_cut_slice = np.squeeze(output_multi_dim)

        processed_pred_roi_cut_slice = process_probability_map(output_pred_roi_cut_slice)

        pred_roi_cut[:, :, i] = processed_pred_roi_cut_slice

    return resize_roi(pred_roi_cut, dx, dy, img.shape)


def predict(data_type, trial_path, net_path, pred_model, input_shape):
    test_data_path = os.path.join(trial_path, '3dData', 'test')
    test_data_list = os.listdir(os.path.join(test_data_path, 'image'))
    if '.DS_Store' in test_data_list:
        test_data_list.remove('.DS_Store')

    for i in test_data_list:
        name = i[0:5]
        img = np.load(os.path.join(test_data_path, 'image', i))
        roi = np.load(os.path.join(test_data_path, 'roi', i))

        if data_type == '3d':
            roi_pred = predict_3d(pred_model, img, roi, input_shape)
        elif data_type == '2d':
            roi_pred = predict_2d(pred_model, img, roi, input_shape)
        else:
            raise RuntimeError('Invalid data type passed to predict')

        np.save(os.path.join(net_path, 'pred', name), roi_pred)
        print('{} roi prediction copmlete'.format(name))







