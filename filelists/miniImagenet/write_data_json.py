"""
Generate json files. Following the standard format in original repository https://github.com/wyharveychen/CloserLookFewShot
"""

import os
import glob

data_path = "/home/CanyuLe/mini-imagenet-original"  # use original mini-imagenet
# data_path = "D:/mini-imagenet-original"

train_path = os.path.join(data_path, "train")
val_path = os.path.join(data_path, "val")
test_path = os.path.join(data_path, "test")

dataset_list = ['base', 'val', 'novel']
savedir = './'

folderlist = []
filelists_flat = {'base':[],'val':[],'novel':[] }
labellists_flat = {'base':[],'val':[],'novel':[] }

label = 0
for i, dire in enumerate([train_path, val_path, test_path]):
    all_sub_dirs = glob.glob(os.path.join(dire, "*"))
    all_sub_dirs = sorted(all_sub_dirs, key=str)
    all_sub_dirs = sorted(all_sub_dirs, key=len)

    all_sub_dirs = [x.replace("\\", "/") for x in all_sub_dirs]

    for pp in all_sub_dirs:
        labelname = pp.split('/')[-1]
        folderlist.append(labelname)

        filelists = glob.glob(os.path.join(pp, "*"))
        filelists = sorted(filelists, key=str)
        filelists = sorted(filelists, key=len)

        filelists = [x.replace("\\", "/") for x in filelists]

        labellists = [label for x in range(len(filelists))]
        if i == 0:
            filelists_flat['base'].extend(filelists)
            labellists_flat['base'].extend(labellists)
        if i ==1:
            filelists_flat['val'].extend(filelists)
            labellists_flat['val'].extend(labellists)
        if i==2:
            filelists_flat['novel'].extend(filelists)
            labellists_flat['novel'].extend(labellists)
        label += 1


print()


for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)
