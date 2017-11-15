from predict import predict_3d
from predict import process_probability_map
from model import model

from utilities import clear_dir

import os
import numpy as np
import scipy.io as sio
from subprocess import call

from scipy.misc import imsave

from post_processing import save_3d
from post_processing import mask_processing

from data import normalize_image

def load_img(mat_path, key):
    mat_dict = sio.loadmat(mat_path)
    img = mat_dict[key]
    return img

def test_img(mat_path, prev_dir, input_dim):
    clear_dir(prev_dir)
    img = load_img(mat_path, 'MTwcorr')
    imsave(os.path.join(prev_dir, 'full_img_proximal.jpg'), img[:, :, 0])
    imsave(os.path.join(prev_dir, 'full_img_distal.jpg'), img[:, :, input_dim[2]-1])

    tst_xcut = 50
    tst_ycut = 50
    tst_cut_img = img[tst_xcut:tst_xcut+input_dim[0], tst_ycut:tst_ycut+input_dim[1], :]

    imsave(os.path.join(prev_dir, 'test_cut_proximal.jpg'), tst_cut_img[:, :, 0])
    imsave(os.path.join(prev_dir, 'test_cut_distal.jpg'), tst_cut_img[:, :, input_dim[2]-1])

def predict_wo_roi(pred_model, img_cut):
    input_multi_dim = np.reshape(img_cut, (1,1,) + img_cut.shape)
    output_multi_dim = pred_model.predict(input_multi_dim)

    output_pred_roi_cut = np.squeeze(output_multi_dim)
    processed_roi_cut = process_probability_map(output_pred_roi_cut)

    return processed_roi_cut

def single_roi_pred(trial_path):
    for i in range(1,4):
        trial_path = os.path.join(trial_path, 'trial{}'.format(i))
        unet_file = os.path.join(trial_path, 'cnn3d', 'unet.hdf5')
        pred_model = model(net='cnn3d',
                           input_shape=(1,)+(48,48,40),
                           init_lr=.00001)
        # init_lr is irrelevant
        pred_model.load_weights(unet_file)

def pipeline(save_dir, mat_path, file, img_key, roi_key, unet_file, input_shape):
    img = load_img(os.path.join(mat_path, file), img_key)
    roi = load_img(os.path.join(mat_path, file), roi_key)

    pred_model = model(net='cnn3d',
                       input_shape=(1,)+input_shape,
                       init_lr=.00001)

    pred_model.load_weights(unet_file)

    # print(pred_model.summary())

    input, raw_output, proc_output, pred = predict_3d(pred_model=pred_model,
                                                      img=img,
                                                      roi=roi,
                                                      input_shape=input_shape,
                                                      pipeline=True)
    #
    # save_3d(img, save_dir, 'img')
    #

    save_path = os.path.join(save_dir, 'image')
    np.save(save_path, img)
    save_path = os.path.join(save_dir, 'cut_input')
    np.save(save_path, input)
    save_path = os.path.join(save_dir, 'raw_output')
    np.save(save_path, raw_output)
    save_path = os.path.join(save_dir, 'processed_output')
    np.save(save_path, proc_output)
    save_path = os.path.join(save_dir, 'resized_output')
    np.save(save_path, pred)


def pipeline_get(base_path):
    local_pred = os.path.join(base_path, 'test', 'pipeline', 'pred')
    local_prev = os.path.join(base_path, 'test', 'pipeline', 'preview')

    clear_dir(local_prev)

    img = np.load(os.path.join(local_pred, 'image.npy'))
    cut_input = np.load(os.path.join(local_pred, 'cut_input.npy'))
    raw_output = np.load(os.path.join(local_pred, 'raw_output.npy'))
    proc_output = np.load(os.path.join(local_pred, 'processed_output.npy'))
    resized_pred = np.load(os.path.join(local_pred, 'resized_output.npy'))

    save_3d(img, local_prev, 'image')
    save_3d(cut_input, local_prev, 'cut_input')
    save_3d(raw_output, local_prev, 'raw_output')
    save_3d(proc_output, local_prev, 'proc_output')
    save_3d(resized_pred, local_prev, 'resized_pred')

    print('done')

def pipeline_local(base_path):
    input_dim = (48, 48, 40)

    trial_path = os.path.join(base_path, 'trial')

    mat_path = os.path.join(base_path, 'corrMTRdata')

    file = 'corrDATA_1165360.mat'

    pred_dir = os.path.join(base_path, 'test', 'pipeline', 'pred')
    clear_dir(pred_dir)

    # test_img(mat_path, prev_dir, input_dim)

    pipeline(save_dir=pred_dir,
             mat_path=mat_path,
             file=file,
             img_key='MTwcorr',
             roi_key='nerveROI',
             unet_file=os.path.join(trial_path, 'trial1', 'cnn3d', 'unet.hdf5'),
             input_shape=input_dim)

    print('done')


if __name__ == '__main__':

    base_path = '/Users/Matthew/Documents/Research/'
    pipeline_local(base_path)

    pipeline_get(base_path)




