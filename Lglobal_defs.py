# -*- coding: utf-8 -*-
"""
@Author: Zhixin Ling
@Description: Part of the project of Data Science: some global definitions. This script should
            include no other scripts in the project.
"""

import os
from enum import IntEnum


join = os.path.join
exists = os.path.exists


def mkdir(dir):
    if not exists(dir):
        os.mkdir(dir)
    return dir


# top directories
FOLDER_DATASETS_RAW = 'datasets_raw'


FOLDER_DATASETS = mkdir('datasets')
FOLDER_LOGS = mkdir('logs')


TR_NAME = 'tr.t'
TE_NAME = 'te.t'


DatasetsA_names = ['a1a', 'a7a', 'a8a', 'pendigits', 'usps', 'cifar10', 'mnist']
class DatasetsA(IntEnum):
    a1a = 0
    a7a = 1
    a8a = 2
    pendigits = 3
    usps = 4
    cifar10 = 5
    mnist = 6


Datasets2_names = ['admissions', 'accidents','diamonds']
class Datasets2(IntEnum):
    admissions = 0
    accidents = 1
    diamonds = 2





if __name__=='__main__':
    #print(DA.filenames[0])
    pass

