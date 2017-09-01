import os
import pickle

from model import model

from train import train

from predict import predict

def segment_2d(shared):
    net = 'cnn2d'
    data_type = '2d'

    init_lr = shared['init_lr']*10

    trial_path = os.path.join(shared['base'], 'trial{}'.format(shared['trial']))
    data_path = os.path.join(trial_path, '3dData')
    net_path = os.path.join(trial_path, net)

    org_input_shape = shared['input_shape']
    input_shape = (org_input_shape[0], org_input_shape[1])

    # Using higher initial learning rate
    train_model = model(net=net,
                        input_shape=(1,)+input_shape,
                        init_lr=init_lr)

    weight_file = 'unet.hdf5'

    history = train(net=net,
                    data_path=data_path,
                    model=train_model,
                    input_shape=input_shape,
                    weight_file=os.path.join(net_path, weight_file),
                    batch_size=shared['batch_size']*40,
                    epoch_count=shared['epoch_count'])

    val_history_file = open(os.path.join(net_path, 'history_val.txt'), 'w')
    for i in history.history['val_dice_coef']:
        val_history_file.write('{}\n'.format(i))

    train_history_file = open(os.path.join(net_path, 'history_train.txt'), 'w')
    for i in history.history['dice_coef']:
        train_history_file.write('{}\n'.format(i))

    pred_model = model(net=net,
                       input_shape=(1,)+input_shape,
                       init_lr=init_lr)

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

    print('Begin trial {} 2d segmentation'.format(shared['trial']))
    segment_2d(shared)

    print('Fin')

























