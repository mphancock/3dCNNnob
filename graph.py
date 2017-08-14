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


if __name__ == '__main__':
    graph()