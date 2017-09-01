import os
from subprocess import call

def clear_directory(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(e)


def clear_directories(base_path, dir_list):
    for dir in dir_list:
        dir_path = os.path.join(base_path, dir)
        clear_directory(dir_path)
        print('The following data path is clear: {}'.format(dir_path))


def clear_dir(dir):
    call(['find', '{}'.format(dir), '-type', 'f', '-name', '*', '-exec', 'rm', '--', '{}', '+'])


