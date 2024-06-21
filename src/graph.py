import os

import numpy as np
import matplotlib.pyplot as plt

def graph():
    dir = '/Users/Matthew/Downloads/3dCNN/'

    graph_1 = 'history_train_dice.txt'
    graph_2 = 'history_val_dice.txt'

    path_1 = os.path.join(dir, graph_1)
    path_2 = os.path.join(dir, graph_2)

    with open(path_1) as f:
        lines_1 = f.read().splitlines()

    with open(path_2) as f:
        lines_2 = f.read().splitlines()

    lines_1 = list(map(float, lines_1))
    lines_2 = list(map(float, lines_2))

    print('training max: {}'.format(max(lines_1)))
    print('validation max: {}'.format(max(lines_2)))

    plt.figure(1)
    plt.subplot(211)
    plt.plot(lines_1, 'b-', linewidth=.5)
    plt.subplot(211)
    plt.plot(lines_2, 'r-', linewidth=.5)

    ave_list_1 = []
    ave_list_2 = []
    x_axis = []

    delta = 50
    for i in range(len(lines_1)//delta):
        values_1 = lines_1[i*delta:(i+1)*delta]
        ave_1 = sum(values_1) / len(values_1)
        ave_list_1.append(ave_1)
        values_2 = lines_2[i*delta:(i+1)*delta]
        ave_2 = sum(values_2) / len(values_2)
        ave_list_2.append(ave_2)

        x_axis.append(i*delta)

    plt.figure(2)
    plt.subplot(211)
    plt.plot(x_axis, ave_list_1, 'b-', linewidth=.5)
    plt.subplot(211)
    plt.plot(x_axis, ave_list_2, 'r-', linewidth=.5)

    p_1 = np.polyfit(x_axis[20:], ave_list_1[20:], 3)
    p_2 = np.polyfit(x_axis[20:], ave_list_2[20:], 3)

    p_list_1 = np.polyval(p_1, x_axis[20:])
    p_list_2 = np.polyval(p_2, x_axis[20:])

    plt.subplot(211)
    plt.plot(x_axis[20:], p_list_1, 'b--', linewidth=.5)
    plt.subplot(211)
    plt.plot(x_axis[20:], p_list_2, 'r--', linewidth=.5)

    p_1 = np.poly1d(p_1)
    p_2 = np.poly1d(p_2)

    p1_roots = (p_1 - .9).r

    print(p1_roots)
    print(p_2(p1_roots[0]))

    plt.show()

def graph_net(trial_path, net, net_num, epoch_count):
    train_file = 'history_train.txt'
    val_file = 'history_val.txt'

    net_dir = os.path.join(trial_path, net)
    train_path = os.path.join(net_dir, train_file)
    val_path = os.path.join(net_dir, val_file)

    with open(train_path) as f:
        train_list = f.read().splitlines()

    with open(val_path) as f:
        val_list = f.read().splitlines()

    subplot = net_num + 220

    x_axis = list()

    for i in range(epoch_count):
        x_axis.append(i)

    plt.subplot(subplot)
    plt.plot(x_axis, train_list, 'b--')
    # plt.subplot(subplot)
    plt.plot(x_axis, val_list, 'r--')
    plt.title(net)




def graph_trial(trial_path):
    plt.figure(1)

    net_num = 1
    net_list = ['cnn3d', 'cnn2d', 'cnnbn', 'cnnwo']
    for net in net_list:
        graph_net(trial_path=trial_path,
                  net=net,
                  net_num=net_num,
                  epoch_count=5000)

        net_num += 1
    plt.show()

if __name__ == '__main__':
    r_path = '/Users/Matthew/Documents/Research/'
    for i in range(1,4):
        trial_path = os.path.join(r_path, 'trial', 'trial{}'.format(i))
        graph_trial(trial_path)
        print('Trial {} graphing complete'.format(i))

