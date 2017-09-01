import os
import math

from keras.callbacks import ModelCheckpoint

from generator import get_generator

from utilities import get_count

gpu_count = 4


def get_callbacks(weight_file):
    model_checkpoint = ModelCheckpoint(weight_file, monitor='loss', save_best_only=True)

    callbacks = [model_checkpoint]
    return callbacks


def train(net, data_path, model, input_shape, weight_file, batch_size, epoch_count):
    model.summary()

    train_data_path = os.path.join(data_path, 'train')
    val_data_path = os.path.join(data_path, 'val')

    train_gen = get_generator(net=net,
                              data_path=train_data_path,
                              input_shape=input_shape,
                              batch_size=batch_size)

    val_gen = get_generator(net=net,
                            data_path=val_data_path,
                            input_shape=input_shape,
                            batch_size=batch_size)

    num_train_img = get_count(os.path.join(train_data_path, 'image'))
    num_val_img = get_count(os.path.join(val_data_path, 'image'))

    print('Number of training volumes: {}'.format(num_train_img))
    print('Number of validation volumes: {}'.format(num_val_img))
    print('Batch size: {}'.format(batch_size))

    if net == 'cnn2d':
        num_train_img = num_train_img * 40
        num_val_img = num_val_img * 40
        print('Number of training images: {}'.format(num_train_img))
        print('Number of validation images: {}'.format(num_val_img))

    train_steps = math.ceil(float(num_train_img / batch_size))
    val_steps = math.ceil(float(num_val_img / batch_size))

    callbacks = get_callbacks(weight_file)

    history = model.fit_generator(generator=train_gen,
                        steps_per_epoch=train_steps,
                        epochs=epoch_count,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        validation_steps=val_steps)

    return history









