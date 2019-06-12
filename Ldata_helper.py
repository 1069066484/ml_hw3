# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Machine Learning: some help functions.
"""

from Lglobal_defs import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import csv
from sklearn.manifold import TSNE
import gzip
from scipy.io import loadmat
import pickle


def posfix_filename(filename, postfix):
    if not filename.endswith(postfix):
        filename += postfix
    return filename


def npfilename(filename):
    return posfix_filename(filename, '.npy')


def pkfilename(filename):
    return posfix_filename(filename, '.pkl')


def csvfilename(filename):
    return posfix_filename(filename, '.csv')


def h5filename(filename):
    return posfix_filename(filename, '.h5')


npfn = npfilename
pkfn = pkfilename
csvfn = csvfilename
h5fn = h5filename


def csvfile2nparr(csvfn, cols=None):
    csvfn = csvfilename(csvfn)
    csvfn = csv.reader(open(csvfn,'r'))
    def read_line(line):
        try:
            # return [float(i) for i in line if cols is None or i in cols]
            # print(cols)
            return [float(line[i]) for i in range(len(line)) if cols is None or i in cols]
        except:
            return None
    m = [read_line(line) for line in csvfn]
    m = [l for l in m if l is not None]
    return np.array(m)


def read_labeled_features(csvfn):
    arr = csvfile2nparr(csvfn)
    data, labels = np.hsplit(arr,[-1])
    labels = labels.reshape(labels.size)
    return [data, labels]



def plt_show_it_data(it_data, xlabel='iterations', ylabel=None, title=None, do_plt_last=True):
    y = it_data
    x = list(range(len(y)))
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    if do_plt_last:
        plt.text(x[-1], y[-1], y[-1])
    plt.show()


def plt_show_scatter(xs, ys, xlabel=None, ylabel=None, title=None):
    colors = ['r', 'y', 'k', 'g', 'b', 'm']
    num2plt = min(len(colors), len(xs))
    for i in range(num2plt):
        plt.scatter(x=xs[i], y=ys[i], c=colors[i], marker='.')
    plt.xlabel('' if xlabel is None else xlabel)
    plt.ylabel('' if ylabel is None else ylabel)
    plt.title('' if title is None else title)
    plt.show()


def non_repeated_random_nums(nums, num):
    num = math.ceil(num)
    nums = np.random.permutation(nums)
    return nums[:num]


def index_split(num, percent1):
    percent1 = math.ceil(num * percent1)
    nums = np.random.permutation(num)
    return [nums[:percent1], nums[percent1:]]


def labeled_data_split(labeled_data, percent_train=0.6):
    np.random.seed(0)
    train_idx, test_idx = index_split(labeled_data[0].shape[0], percent_train)
    train_ld = [labeled_data[0][train_idx], labeled_data[1][train_idx]]
    test_ld = [labeled_data[0][test_idx], labeled_data[1][test_idx]]
    return [train_ld, test_ld]


def rand_arr_selection(arr, num):
    nonrep_rand_nums = non_repeated_random_nums(arr.shape[0], num)
    return [arr[nonrep_rand_nums], nonrep_rand_nums]


def labels2one_hot(labels):
    labels = np.array(labels, dtype=np.int)
    if len(labels.shape) == 1:
        minl = np.min(labels)
        labels -= minl
        maxl = np.max(labels) + 1
        r = range(maxl)
        return np.array([[1 if i==j else 0 for i in r] for j in labels])
    return labels


def shuffle_labeled_data(dl):
    data, labels = dl
    a = np.arange(labels.shape[0])
    np.random.seed(0)
    np.random.shuffle(a)
    return [data[a], labels[a]]



def _test_labels_one_hot():
    a = np.array([2,1,0,0,0,2,1,1,1])
    print(labels2one_hot(a))


def _read_dataset_mini_test():
    p = r'G:\f\SJTUstudy\G3_SEMESTER2\machine_learning\prj\Bdataset\fer2013\t.csv'
    #print(os.path.exists(p))
    #exit(0)
    f = csv.reader(open(p))
    tr_labels = []
    te_labels = []
    tr_data = []
    te_data = []
    for label, data1, trte1  in f:
        data1 = data1.split(' ')
        if trte1.startswith('Tr'):
            #print(len(data1))
            tr_labels.append(int(label))
            tr_data.append([int(i) for i in data1])
        elif trte1.startswith('Te'):
            #print(len(data1))
            te_labels.append(int(label))
            te_data.append([int(i) for i in data1])
    tr_data = np.array(tr_data)
    print(tr_data[1])
    te_data = np.array(te_data)
    tr_labels = np.array(tr_labels)
    te_labels = np.array(te_labels)
    for a in [tr_data, te_data, tr_labels, te_labels]:
        print(a.shape)


def raw_axa2dl(fn):
    labels = []
    data = []
    init_arr = np.zeros(123)
    for l in open(fn):
        # print(l.split())
        l = l.split()
        data.append(init_arr.copy())
        labels.append(0 if l[0] == '-1' else 1)
        for str in l[1:]:
            data[-1][int(str[:-2])-1] = 1
    return [np.array(data, dtype=np.int8), np.array(labels, dtype=np.int8)]


def raw_pendigits2dl(fn):
    labels = []
    data = []
    init_arr = np.zeros(16)
    for l in open(fn):
        l = l.split()
        data.append(init_arr.copy())
        labels.append(int(l[0]))
        for str in l[1:]:
            pos, val = str.split(':')
            data[-1][int(pos)-1] = float(val)/100.0
    return [np.array(data, dtype=np.float32), np.array(labels, dtype=np.int8)]


def raw_usps2dl(fn):
    labels = []
    data = []
    for l in open(fn):
        l = l.split()
        labels.append(int(l[0]))
        data.append([-float(str.split(':')[1]) for str in l[1:]])
    return [np.array(data, dtype=np.float32), np.array(labels, dtype=np.int8)]


def read_mnist_dls(path):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(path, one_hot=False)
    return [[mnist.train.images, mnist.train.labels],[mnist.test.images,mnist.test.labels]]


def read_cifar10_dls(folder):
    def read_batch_dls(path):
        with open(path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return [dict[b'data'], dict[b'labels']]
    tr_d = []
    tr_l = []
    for path in ['data_batch_' + str(i) for i in range(1,6)]:
        tr_d_b, tr_l_b = read_batch_dls(join(folder, path))
        tr_d.append(tr_d_b)
        tr_l += tr_l_b
    tr_d = np.vstack(tr_d)
    tr_l = np.array(tr_l, dtype=np.uint8)
    tr_d = (tr_d/256.0).astype(np.float32)
    te_d, te_l = read_batch_dls(join(folder, 'test_batch'))
    te_d = (te_d/256.0).astype(np.float32)
    te_l = np.array(te_l, dtype=np.uint8)
    return [[tr_d, tr_l], [te_d, te_l]]


def read_dataset_A(datasetsA):
    """
    return [[train_data, train_labels], [test_data, test_labels]]
    """
    path = join(FOLDER_DATASETS, DatasetsA_names[datasetsA])
    path_tr_d = join(path, npfn('tr_d'))
    path_tr_l = join(path, npfn('tr_l'))
    path_te_d = join(path, npfn('te_d'))
    path_te_l = join(path, npfn('te_l'))
    ld = np.load
    if exists(path):
        return [[ld(path_tr_d), ld(path_tr_l)],
                [ld(path_te_d), ld(path_te_l)]]
    folder = join(FOLDER_DATASETS_RAW, DatasetsA_names[datasetsA])
    if datasetsA == DatasetsA.mnist:
        tr_dl, te_dl = read_mnist_dls(folder)
    elif datasetsA == DatasetsA.cifar10:
        tr_dl, te_dl = read_cifar10_dls(folder)
    else:
        raw2dls = [raw_axa2dl,raw_axa2dl,raw_axa2dl,raw_pendigits2dl,raw_usps2dl]
        tr_dl = raw2dls[datasetsA](join(folder, TR_NAME))
        te_dl = raw2dls[datasetsA](join(folder, TE_NAME))
    path = mkdir(path)
    np.save(path_tr_d, tr_dl[0])
    np.save(path_tr_l, tr_dl[1])
    np.save(path_te_d, te_dl[0])
    np.save(path_te_l, te_dl[1])
    return [tr_dl, te_dl]


def read_dataset_2(datasets2):
    path = join(FOLDER_DATASETS, Datasets2_names[datasets2])
    path_data = npfn(join(path, Datasets2_names[datasets2]))
    if exists(path):
        return np.load(path_data)
    path = mkdir(path)
    folder = join(FOLDER_DATASETS_RAW, Datasets2_names[datasets2])
    path_csv = csvfn(join(folder, Datasets2_names[datasets2]))
    if datasets2 == Datasets2.admissions or datasets2 == Datasets2.diamonds:
        data = csvfile2nparr(path_csv).astype(np.float32)[:,1:]
    elif datasets2 == Datasets2.accidents:
        data = csvfile2nparr(path_csv, [7,9,10,11,12]).astype(np.float32)
    np.save(path_data, data)
    return data


def _test_read_dataset_A():
    for ds in [
    DatasetsA.a1a, DatasetsA.a7a, DatasetsA.a8a, 
            DatasetsA.pendigits, DatasetsA.usps,
            DatasetsA.cifar10
            ]:
        tr_dl, te_dl = read_dataset_A(ds)
        print('\n',DatasetsA_names[ds])
        print(tr_dl[0].shape, te_dl[0].shape, te_dl[0].shape[0] + tr_dl[0].shape[0])
    """
    (1605, 123)
    (1605,)
    (30956, 123)
    (30956,)
    """


def _test_read_dataset_2():
    data = read_dataset_2(Datasets2.diamonds)
    print(data.shape)


if __name__ == '__main__':
    _test_read_dataset_A()
    # print(__file__)
    # print(exists('./datasets_raw'))