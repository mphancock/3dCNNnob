import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def elastic_transform(image, roi, alpha=100, sigma=3, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))

    indices = np.reshape(y+dy, (-1,1)), np.reshape(x+dx, (-1,1)), np.reshape(z+dz, (-1,1))

    out_img = map_coordinates(image, indices, order=1).reshape(shape)
    out_roi = map_coordinates(roi, indices, order=1).reshape(shape)

    return out_img, out_roi


def elastic_transform_2d(image, roi, alpha=100, sigma=3, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

    indices = np.reshape(y+dy, (-1,1)), np.reshape(x+dx, (-1,1))

    out_img = map_coordinates(image, indices, order=1).reshape(shape)
    out_roi = map_coordinates(roi, indices, order=1).reshape(shape)

    return out_img, out_roi
