import os
import pickle

from train import train

from predict import predict

from model import model

def segment_bn(shared):
    net = 'cnnbn'
    data_type = '3d'

    trial_path = os.path.join(shared['base'], 'trial', 'trial{}'.format(shared['trial']))
    data_path = os.path.join(trial_path, '3dData')
    net_path = os.path.join(trial_path, net)

    input_shape = shared['input_shape']

    # Using increased initial learning rate
    train_model = model(net=net,
                        input_shape=(1,)+input_shape,
                        init_lr=shared['init_lr'])

    weight_file = 'unet.hdf5'

    history = train(net=net,
                    data_path=data_path,
                    model=train_model,
                    input_shape=input_shape,
                    weight_file=os.path.join(net_path, weight_file),
                    batch_size=shared['batch_size'],
                    epoch_count=shared['epoch_count'])

    train_hist_file = open(os.path.join(net_path, 'history_val.txt'), 'w')
    for i in history.history['val_dice_coef']:
        train_hist_file.write('{}\n'.format(i))

    val_hist_file = open(os.path.join(net_path, 'history_train.txt'), 'w')
    for i in history.history['dice_coef']:
        val_hist_file.write('{}\n'.format(i))

    pred_model = model(net=net,
                       input_shape=(1,)+input_shape,
                       init_lr=shared['init_lr'])

    pred_model.load_weights(os.path.join(net_path, weight_file))

    print('Training complete and weights saved')

    predict(data_type=data_type,
            trial_path=trial_path,
            net_path=net_path,
            pred_model=pred_model,
            input_shape=input_shape)


if __name__ == '__main__':
    with open('shared1.pickle', 'rb') as f:
        shared = pickle.load(f)

    segment_bn(shared)

    print('Fin')
