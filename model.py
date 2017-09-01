import numpy as np

from keras import backend as K

from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.optimizers import SGD

K.set_image_dim_ordering('th')

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def compute_level_output_shape(filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    if depth != 0:
        output_image_shape = np.divide(image_shape, np.multiply(pool_size, depth)).tolist()
    else:
        output_image_shape = image_shape
    return tuple([None, filters] + [int(x) for x in output_image_shape])


def get_upconv(depth, nb_filters, pool_size, image_shape, kernel_size=(2, 2, 2), strides=(2, 2, 2),
               deconvolution=False):
    if deconvolution:
        try:
            from keras_contrib.layers import Deconvolution3D
        except ImportError:
            raise ImportError("Install keras_contrib in order to use deconvolution. Otherwise set deconvolution=False.")

        return Deconvolution3D(filters=nb_filters, kernel_size=kernel_size,
                               output_shape=compute_level_output_shape(filters=nb_filters, depth=depth,
                                                                       pool_size=pool_size, image_shape=image_shape),
                               strides=strides, input_shape=compute_level_output_shape(filters=nb_filters,
                                                                                       depth=depth+1,
                                                                                       pool_size=pool_size,
                                                                                       image_shape=image_shape))
    else:
        return UpSampling3D(size=pool_size)


def unet_model_3d(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """

    print('input shape: {}'.format(input_shape))
    inputs = Input(input_shape)
    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(inputs)
    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(pool1)
    conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(pool2)
    conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

    conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(pool3)
    conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv4)

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                     nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
    up5 = concatenate([up5, conv3], axis=1)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv5)

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=int(256/downsize_filters_factor), image_shape=input_shape[-3:])(conv5)
    up6 = concatenate([up6, conv2], axis=1)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
    up7 = concatenate([up7, conv1], axis=1)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',
                   padding='same')(conv7)

    conv8 = Conv3D(n_labels, (1, 1, 1))(conv7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    sgd = SGD(lr=initial_learning_rate, momentum=0.99, decay=0.0, nesterov=False)

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

    # model = make_parallel(model, gpu_count)

    return model


def unet_model_2d(input_shape, pool_size=(2,2), n_labels=1, initial_learning_rate=.00001):

    input_layer = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), padding="same", activation="relu")(input_layer)
    conv1 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size)(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", activation="relu")(pool1)
    conv2 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", activation="relu")(pool2)
    conv3 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", activation="relu")(pool3)
    conv4 = Conv2D(512, (3, 3), padding="same", activation="relu")(conv4)

    up5 = UpSampling2D(size=pool_size)(conv4)
    up5 = concatenate([up5, conv3], axis=1)

    conv5 = Conv2D(256, (3, 3), padding="same", activation="relu")(up5)
    conv5 = Conv2D(256, (3, 3), padding="same", activation="relu")(conv5)

    up6 = UpSampling2D(size=pool_size)(conv5)
    up6 = concatenate([up6, conv2], axis=1)

    conv6 = Conv2D(128, (3, 3), padding="same", activation="relu")(up6)
    conv6 = Conv2D(128, (3, 3), padding="same", activation="relu")(conv6)

    up7 = UpSampling2D(size=pool_size)(conv6)
    up7 = concatenate([up7, conv1], axis=1)

    conv7 = Conv2D(64, (3, 3), padding="same", activation="relu")(up7)
    conv7 = Conv2D(64, (3, 3), padding="same", activation="relu")(conv7)

    conv8 = Conv2D(n_labels, (1, 1))(conv7)
    act = Activation('sigmoid')(conv8)

    model = Model(inputs=input_layer, outputs=act)

    sgd = SGD(lr=initial_learning_rate, momentum=0.99, decay=0.0, nesterov=False)

    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

    return model


def unet_model_bn(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
                  initial_learning_rate=0.00001, deconvolution=False):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). 
    :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """

    print('input shape: {}'.format(input_shape))
    inputs = Input(input_shape)
    conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3),
                   padding='same')(inputs)
    bn1 = BatchNormalization(axis=1)(conv1)
    act1 = Activation('relu')(bn1)

    conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3),
                   padding='same')(act1)
    bn1 = BatchNormalization(axis=1)(conv1)
    act1 = Activation('relu')(bn1)

    pool1 = MaxPooling3D(pool_size=pool_size)(act1)

    conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3),
                   padding='same')(pool1)
    bn2 = BatchNormalization(axis=1)(conv2)
    act2 = Activation('relu')(bn2)

    conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3),
                   padding='same')(act2)
    bn2 = BatchNormalization(axis=1)(conv2)
    act2 = Activation('relu')(bn2)

    pool2 = MaxPooling3D(pool_size=pool_size)(act2)

    conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3),
                   padding='same')(pool2)
    bn3 = BatchNormalization(axis=1)(conv3)
    act3 = Activation('relu')(bn3)

    conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3),
                   padding='same')(act3)
    bn3 = BatchNormalization(axis=1)(conv3)
    act3 = Activation('relu')(bn3)

    pool3 = MaxPooling3D(pool_size=pool_size)(act3)

    conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3),
                   padding='same')(pool3)
    bn4 = BatchNormalization(axis=1)(conv4)
    act4 = Activation('relu')(bn4)

    conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3),
                   padding='same')(act4)
    bn4 = BatchNormalization(axis=1)(conv4)
    act4 = Activation('relu')(bn4)

    up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
                     nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(act4)
    up5 = concatenate([up5, act3], axis=1)

    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), padding='same')(up5)
    bn5 = BatchNormalization(axis=1)(conv5)
    act5 = Activation('relu')(bn5)

    conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3),
                   padding='same')(act5)
    bn5 = BatchNormalization(axis=1)(conv5)
    act5 = Activation('relu')(bn5)

    up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
                     nb_filters=int(256/downsize_filters_factor), image_shape=input_shape[-3:])(act5)
    up6 = concatenate([up6, act2], axis=1)

    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), padding='same')(up6)
    bn6 = BatchNormalization(axis=1)(conv6)
    act6 = Activation('relu')(bn6)

    conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3),
                   padding='same')(act6)
    bn6 = BatchNormalization(axis=1)(conv6)
    act6 = Activation('relu')(bn6)

    up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
                     nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(act6)
    up7 = concatenate([up7, act1], axis=1)

    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), padding='same')(up7)
    bn7 = BatchNormalization(axis=1)(conv7)
    act7 = Activation('relu')(bn7)

    conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3),
                   padding='same')(act7)
    bn7 = BatchNormalization(axis=1)(conv7)
    act7 = Activation('relu')(bn7)

    conv8 = Conv3D(n_labels, (1, 1, 1))(act7)
    act = Activation('sigmoid')(conv8)
    model = Model(inputs=inputs, outputs=act)

    sgd = SGD(lr=initial_learning_rate, momentum=0.99, decay=0.0, nesterov=False)

    # model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=sgd, loss=dice_coef_loss, metrics=[dice_coef])

    # model = make_parallel(model, gpu_count)

    return model


def model(net, input_shape, init_lr):
    if net == 'cnn3d':
        return unet_model_3d(input_shape=input_shape,
                             initial_learning_rate=init_lr)
    elif net == 'cnn2d':
        return unet_model_2d(input_shape=input_shape,
                             initial_learning_rate=init_lr)
    elif net == 'cnnbn':
        return unet_model_bn(input_shape=input_shape,
                                initial_learning_rate=init_lr)
    elif net == 'cnnwo':
        return unet_model_3d(input_shape=input_shape,
                             initial_learning_rate=init_lr)
    else:
        raise RuntimeError('Invalid model type passed to model')





