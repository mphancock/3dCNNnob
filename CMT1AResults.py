import pickle
import os
import math

from subprocess import call
from random import shuffle

from split_data import process_file
from utilities import clear_dir

# set the default parameters
def set_params(testnum):
    shared = {}
    shared['base'] = '/scratch/hancocmp/'
    shared['mat_dir'] = 'biasFieldCorrData'
#    shared['trial'] = 1
    shared['image_shape'] = (256,256,40)
    shared['input_shape'] = (48,48,40)
    shared['init_lr'] = .00001
    shared['batch_size'] = 5
    shared['epoch_count'] = 5000
    shared['def_var'] = (50,3)

    if testnum == 1:
        shared['batch_size'] = 1
    elif testnum == 2:
        shared['batch_size'] = 3
# trial 3 is default batch_size = 5
    elif testnum == 4:
        shared['def_var'] = [50,3]
# trial 5 is default def_var = [100,3]
    elif testnum == 6:
        shared['def_var'] = [150,3]
# trial 7 is default input_shape = (x,x,40)
    elif testnum == 8:
        shared['input_shape'] = (48,48,40)

    return shared


def check_path(path):
    if not os.path.isdir(path):
        raise IOError('{} path does not exist!'.format(path))


def splitCMT1AData(mat_path, data_path):
    CMT1APatients = {}
    CMT1APatients['2001'] = [False, '20010']
    CMT1APatients['2002'] = [False, '20020', '20022']
    CMT1APatients['2003'] = [False, '20030']
    CMT1APatients['2007'] = [False, '20070']
    CMT1APatients['2009'] = [False, '20090', '1162010']
    CMT1APatients['2015'] = [False, '20150']
    CMT1APatients['2016'] = [False, '20160']
    CMT1APatients['2018'] = [False, '20180']
    CMT1APatients['2020'] = [False, '20200', '1150560', '1279830']
    CMT1APatients['2021'] = [False, '20210', '1150570', '1279360']
    CMT1APatients['2031'] = [False, '20310', '2131220', '2161250']
    CMT1APatients['2042'] = [False, '2146490']

    files = os.listdir(mat_path)
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    print('original number of files: {}'.format(len(files)))

    nonTestData = []
    for file in files:
        name = file[9:14]
        hit = False


        # python 2.7 syntax: range(x,y)
        # python 3.5 syntax: list(range(x,y))
        if file[14] in [str(n) for n in range(0, 10)]:
            name = file[9:16]
            hit = True

        # python 2.7 syntax: dict.iteritems()
        # python 3.x syntax: dict.items()
        for subjectIDKey, fileID in CMT1APatients.iteritems():  # for name, age in list.items():  (for Python 3.x)
            if name in fileID and (CMT1APatients[subjectIDKey])[0] == False:
                subjectID = subjectIDKey
                testDataPath = os.path.join(data_path, 'test')

                process_file(file=file,
                         mat_path=mat_path,
                         data_path=testDataPath)

                (CMT1APatients[subjectID])[0] = True
            else:
                nonTestData.append(name)

    #removes duplicate file name
    nonTestData = list(set(nonTestData))
    shuffle(files)

    splitPct = .8
    numTrainMRI = math.floor(len(nonTestData)*splitPct)
    count = 0

    trainDataPath = os.path.join(data_path, 'train')
    valDataPath = os.path.join(data_path, 'val')

    for name in nonTestData:
        for file in files:
            if name in file:
                if count <= numTrainMRI:
                    process_file(file=file,
                                 mat_path=mat_path,
                                 data_path=trainDataPath)
                else:
                    process_file(file=file,
                                 mat_path=mat_path,
                                 data_path=valDataPath)
            else:
                print('{} not found in MAT file list')
        count += 1


    print('All .MAT data saved to the following directory: {}'.format(data_path))


def CMT1A_Nets():
    shared = set_params(testnum=0)

    base_path = os.path.join(shared['base'],'CMT1AResults')

    check_path(base_path)

    # make sure this line works?
    clear_dir(base_path)

    mat_path = os.path.join(shared['base'], shared['mat_dir'])
    data_path = os.path.join(base_path, '3dData')

    # split_data(mat_dir, data_path)
    splitCMT1AData(mat_path=mat_path,
                   data_path=data_path)

    for testnum in range(1,9):
        params = set_params(testnum)
        params['trialnum'] = testnum

        with open('shared{}.pickle'.format(testnum), 'wb') as f:
            pickle.dump(params, f)

        call(['sbatch', 'segCMT1A{}.slurm'.format(testnum)])
        print('CMT1A segmentation trial {} has begun using bn CNN with elastic deformation')


if __name__ == '__main__':
    CMT1A_Nets()
    print('Fin')

