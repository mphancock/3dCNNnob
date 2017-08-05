import os

import dill

from scipy.misc import imsave

from train import train

from save_data import save_data

from clear_directories import clear_data_directories

from predict import evaluate_model
from predict import predict

from model import unet_model_3d

'''
    Need the following data structure: 
    3dData: 
        -->test: 
            -->image: 
            -->roi: 
            -->pred:
            -->aff: 
        -->train: 
            -->image:
            -->roi: 
        -->val: 
            -->image: 
            -->roi: 
'''

def segment(type):
    if type == 'local':
        mat_path = '/Users/Matthew/Documents/Research/corrMTRdata'
        data_path = '/Users/Matthew/Documents/Research/3dData'
    elif type == 'cluster':
        mat_path = '/scratch/hancocmp/corrMTRdata/'
        data_path = '/scratch/hancocmp/3dData/'
    else:
        raise RuntimeError("Invalid 'type' parameter passed")

    check_file(mat_path)
    check_file(data_path)

    clear_data_directories(data_path)
    print('directories clear')

    save_data(mat_path, data_path)
    print('all data saved')

    img_shape = (256,256,40)
    affine_shape = (48,48,40)
    input_shape = (48,48,48)

    #input fed into fist layer: (None, 1, 48, 48, 64)
    model_input = (1,48,48,48)

    train_model = unet_model_3d(model_input,
                          downsize_filters_factor=1,
                          initial_learning_rate=.00001)

    history = train(data_path, train_model, input_shape, batch_size=4, epoch_count=4000)

    history_file = open('history_val_dice.txt', 'w')
    for i in history.history['val_dice_coef']:
        history_file.write('{}\n'.format(i))

    history_file_2 = open('history_train_dice.txt', 'w')
    for i in history.history['dice_coef']:
        history_file_2.write('{}\n'.format(i))

    pred_model = unet_model_3d(model_input,
                          downsize_filters_factor=1,
                          initial_learning_rate=.00001)

    pred_model.load_weights('unet.hdf5')

    print('training complete')
    print('weights saved')

    # scores = evaluate_model(pred_model)
    predict(pred_model, data_path, img_shape, input_shape, affine_shape)

    # print(scores)

    test_dir = os.path.join(data_path, 'test')
    test_file_list = os.listdir(test_dir)

    if '.DS_Store' in test_file_list:
        test_file_list.remove('.DS_Store')

    print('roi prediction complete')


def save_roi(dir, file, roi):
    for i in range(roi.shape[2]):
        name = '{}slice{}.jpg'.format(file, i)
        imsave(os.path.join(dir, name), roi[:, :, i])


def check_file(file):
    if not os.path.isdir(file):
        raise IOError('{} path does not exist!'.format(file))


if __name__ == '__main__':
    type = 'cluster'
    segment(type)
    print('fin')

