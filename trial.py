import pickle
import os
from subprocess import call

from split_data import split_data
from clear_directories import clear_dir


def set_params():
    shared = {}
    shared['base'] = '/scratch/hancocmp/'
    shared['mat_dir'] = 'corrMTRdata'
    shared['trial'] = 1
    shared['image_shape'] = (256,256,40)
    shared['input_shape'] = (48,48,40)
    shared['init_lr'] = .00001
    shared['batch_size'] = 5
    shared['epoch_count'] = 5000

    return shared


def check_path(path):
    if not os.path.isdir(path):
        raise IOError('{} path does not exist!'.format(path))


def run_trial(shared):
    for k, v in shared.items():
        print(k, v)

    base_path = os.path.join(shared['base'], 'trial{}'.format(shared['trial']))
    check_path(base_path)
    clear_dir(base_path)

    mat_dir = os.path.join(shared['base'], shared['mat_dir'])
    data_path = os.path.join(base_path, '3dData')

    split_data(mat_dir, data_path)

    call(['sbatch', 'cnn3d.slurm'])
    print('3d segmentation with non linear augmentation has begun')

    call(['sbatch', 'cnn2d.slurm'])
    print('2d segmentation with non linear augmentation has begun')

    call(['sbatch', 'cnnbn.slurm'])
    print('3d batch normalizing segmentation and non linear augmentation has begun')

    call(['sbatch', 'cnnwo.slurm'])
    print(['3d segmentation without linear/non linear augmentation has begun'])


if __name__ == '__main__':
    shared = set_params()

    with open('shared1.pickle', 'wb') as f:
        pickle.dump(shared, f)

    run_trial(shared)
    print('Fin')

