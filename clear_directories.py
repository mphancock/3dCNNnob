import os

def clear_directory(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(e)


def clear_data_directories(base_path):
    directories = ['test', 'val', 'train']

    for i in directories:
        image_path = os.path.join(base_path, i, 'image')
        roi_path = os.path.join(base_path, i, 'roi')

        clear_directory(image_path)
        clear_directory(roi_path)

        if i == 'test':
            pred_path = os.path.join(base_path, i, 'pred')
            clear_directory(pred_path)


if __name__ == '__main__':
    clear_data_directories('/Users/Matthew/Documents/Research/3dData')
