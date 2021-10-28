import math
import os

from matplotlib import pyplot as plt
import json
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy.spatial.distance as dist
from collections import defaultdict
import numpy as np
import sys
from dtw import *
from pathlib import Path


def generates_distances2D(x, y):
    '''
    Generates the distance matrix used with Dynamic Time Warping
    '''
    # Creates the helper matrix
    helper_matrix = np.ones((y.shape[0], x.shape[0]))

    # Generates (y.shape[0]) copies of each value in original_drawing
    x_all = (x[:, 0] * helper_matrix)
    y_all = (x[:, 1] * helper_matrix)

    # returns all the distances between original points and generated ones
    return (np.sqrt((x_all.T - y[:, 0]) ** 2 + (y_all.T - y[:, 1]) ** 2))


def generates_accumulated_cost_p(x, y, distances):
    accumulated_cost = numpy.zeros((len(x), len(y)))

    accumulated_cost[0, 0] = distances[0, 0]

    for i in range(1, len(x)):
        accumulated_cost[i, 0] = distances[i, 0] + accumulated_cost[i - 1, 0]

    for j in range(1, len(y)):
        accumulated_cost[0, j] = distances[0, j] + accumulated_cost[0, j - 1]

    for i in range(1, len(x)):
        for j in range(1, len(y)):
            accumulated_cost[i, j] = min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                         accumulated_cost[i, j - 1]) + distances[i, j]

    return accumulated_cost


def path_cost_p(x, y, accumulated_cost, distances):
    path = [[len(x) - 1, len(y) - 1]]

    cost = 0

    i = len(x) - 1
    j = len(y) - 1

    while i > 0 or j > 0:
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if accumulated_cost[i - 1, j] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                 accumulated_cost[i, j - 1]):
                i = i - 1
            elif accumulated_cost[i, j - 1] == min(accumulated_cost[i - 1, j - 1], accumulated_cost[i - 1, j],
                                                   accumulated_cost[i, j - 1]):
                j = j - 1
            else:
                i = i - 1
                j = j - 1
        path.append([i, j])
    for [x, y] in path:
        cost = cost + distances[x, y]
    return path, cost


def get_traj(name, root_path='Datas/'):
    folder = Path(root_path + name).iterdir()
    trajs = []
    for path in folder:
        name = Path(path).name[:-5]
        with open(path, 'r') as json_file:
            l = []
            data = json.load(json_file)
            for i in range(len(data['traj'])):
                _x = round(data['traj'][i]['x'], 4)
                _z = round(data['traj'][i]['z'], 4)
                l.append((_x, _z))
            traj = np.array(l)
            trajs.append((traj, name))
    return trajs


def comparison(t1, t2, name, save_path=""):
    costs = []
    str_l = []
    path = save_path + name + r"\\"
    if path != "":
        if not os.path.exists(path):
            os.makedirs(path)
    for x in t1:
        for y in t2:
            cost, result = compute_dtw(x[0], y[0], x[1], y[1], path)
            str_l.append(str(result))
            print(result)
            costs.append(cost)

    str_l.append(str(f"{name} mean cost : {np.mean(costs)}"))
    print(f"{name} mean cost : {np.mean(costs)}")
    with open(path+name+".txt",'a') as f:
        for i in str_l:
            f.write(i+'\n')


def compute_dtw(x, y, x_name, y_name, save_path="", show=False):
    x_traj, y_traj = x, y
    title_name = x_name + ' & ' + y_name

    dist = generates_distances2D(x_traj, y_traj)
    accumulated_cost = generates_accumulated_cost_p(x_traj, y_traj, dist)
    dtw_path, cost = path_cost_p(x_traj, y_traj, accumulated_cost=accumulated_cost, distances=dist)
    dtw_path = numpy.array(dtw_path)
    plt.figure(figsize=(20, 30))


    plt.title(title_name)

    for [x, y] in dtw_path:
        plt.plot([x_traj[x, 0], y_traj[y, 0]], [x_traj[x, 1], y_traj[y, 1]], c='C7', markersize=0.1)

    plt.plot(x_traj[:, 0], x_traj[:, 1], 'ro-', label=x_name[:-2], markersize=1)
    plt.plot(y_traj[:, 0], y_traj[:, 1], 'g^-', label=y_name[:-2], markersize=1)
    plt.legend()

    result = title_name + ' cost :'+ str(cost)
    fig = plt.gcf()
    fig.savefig(os.path.join(save_path + title_name + ".png"), dpi=300)
    plt.clf()
    plt.close()
    return cost, result


ppo = "PPO"
soft_lr = "SoftLR"
hard_lr = "HardLR"
soft_break = "SoftBreak"
hard_break = "HardBreak"
demo_lr = "LRDemo"
demo_break = "BreakDemo"

save_path = r"R:\DTW\Img\\"

if __name__ == "__main__":
    ppo = get_traj(ppo)

    soft_lr = get_traj(soft_lr)
    hard_lr = get_traj(hard_lr)

    demo_lr = get_traj(demo_lr)
    demo_break = get_traj(demo_break)

    soft_break = get_traj(soft_break)
    hard_break = get_traj(hard_break)
    #comparison(ppo, soft_lr, "PPO, SoftLR", save_path)
    comparison(ppo, hard_lr, "PPO, HardLR", save_path)

    comparison(demo_lr, soft_lr, "LRDemo, SoftLR", save_path)
    comparison(demo_lr, hard_lr, "LRDemo, HardLR", save_path)

    comparison(ppo, soft_break, "PPO, SoftBreak", save_path)
    comparison(ppo, hard_break, "PPO, HardBreak", save_path)

    comparison(demo_break, soft_break, "BreakDemo, SoftBreak", save_path)
    comparison(demo_break, hard_break, "BreakDemo, HardBreak", save_path)
