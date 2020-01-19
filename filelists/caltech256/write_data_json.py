import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random


data_path = "/home/CanyuLe/caltech256"
# data_path = "D:/datasets_for_ML/CUB_200_2011/images"


savedir = './'
dataset_list = ['base','val','novel']


folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

# parse split files
split_file_paths = ["./split_files/train_classes.txt", "./split_files/val_classes.txt", "./split_files/test_classes.txt"]
dataset_split = {}
for dataset_name, split_file_path in zip(dataset_list, split_file_paths):
    dataset_split[dataset_name] = []
    with open(split_file_path) as f:
        all_lines = f.read().splitlines()
        for line in all_lines:
            label = label_dict[line]
            dataset_split[dataset_name].append(label)

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = os.path.join(data_path, folder)
    classfile_list_all.append([os.path.join(folder_path, cf) for cf in listdir(folder_path) if (isfile(os.path.join(folder_path,cf)) and cf[0] != '.')])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        classfile_list = [x.replace("\\", "/") for x in classfile_list]

        if 'base' in dataset:
            if i in dataset_split[dataset]:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset:
            if i in dataset_split[dataset]:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset:
            if i in dataset_split[dataset]:
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
