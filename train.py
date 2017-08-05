import os
import math

from keras.callbacks import ModelCheckpoint

from generator import get_generator
from utilities import get_count
from multi_gpu import make_parallel


gpu_count = 4

def train(data_path, model, input_shape, batch_size=1, epoch_count=10000):
    train_path = os.path.join(data_path, 'train')
    train_generator = get_generator(train_path, batch_size, input_shape)

    val_path = os.path.join(data_path, 'val')
    val_generator = get_generator(val_path, batch_size, input_shape)

    # model = make_parallel(model, gpu_count=4)

    model.summary()

    num_train_images = get_count(os.path.join(train_path, 'image'))
    print('Number of training images: {}'.format(num_train_images))
    train_steps_per_epoch = math.ceil(float(num_train_images) / batch_size)

    num_val_images = get_count(os.path.join(val_path, 'image'))
    print('Number of validation images: {}'.format(num_val_images))
    val_steps_per_epoch = math.ceil(float(num_val_images) / batch_size)

    callbacks = get_callbacks()

    history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epoch_count,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch)

    return history

def get_callbacks():
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    callbacks = [model_checkpoint]
    return callbacks



