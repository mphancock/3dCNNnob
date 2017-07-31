from generator import get_generator

from utilities import get_count
from utilities import sum_border

from data import affine_registration
from data import get_stack
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


def predict(pred_model, data_path, img_shape, model_shape, affine_shape):
    test_dir = os.path.join(data_path, 'test')
    test_files = os.listdir(test_dir)

    if '.DS_Store' in test_files:
        test_files.remove('.DS_Store')

    img_stack, roi_stack = get_stack(data_path, img_shape)

    for i in test_files:
        name = i[9:14]

        img = np.load(os.path.join(test_dir, i))
        #img has shape: (256,256,40)
        affine_img, dx, dy = affine_registration(affine_shape, img_stack, roi_stack, img)
        #prediction img now has shape: (32,32,40)
        resized_affine_img = zero_pad(affine_img, shape=model_shape)
        #prediction img now has shape: (32,32,48) == input model shape
        #zero-pad will add xy zero matrices in both the positive and negative z-direction
        pred_roi  = pred_model.predict(resized_affine_img)
        #prediction roi now has shape: (32,32,48)
        resized_pred_roi = resize_roi(pred_roi, dx, dy, img_shape)
        #prediction roi now has shape: (256,256,48)
        resized_pred_roi = resized_pred_roi[:, :, 4:44]
        #prediction roi now has shape: (256,256,40) == img shape

        pred_save_path = os.path.join(test_dir, 'pred', name)

        border_sum = sum_border(resized_pred_roi)

        if border_sum == 0:
            np.save(pred_save_path, resized_pred_roi)
            print('ndarray saved -- name: {}\tshape: {}\tpath: {}'.format(name, resized_pred_roi.shape, pred_save_path))
        else:
            print('affine failure -- name: {}\tborder sum: {}'.format(name, border_sum))


