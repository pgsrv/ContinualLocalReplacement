import numpy as np
import torch
import torch.nn as nn
import time
import os
import glob

import configs
from data.datamgr import SimpleDataManager, JigsawDataManger, SetDataManager
from methods.jigsaw import Jigsaw
from methods.imprint_jigsaw import ImprintJigsaw
from io_utils import parse_args, get_resume_file, get_best_file , get_assigned_file


if __name__ == '__main__':
    params = parse_args('test')

    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu

    test_file = configs.data_dir[params.dataset] + 'novel.json'

    record_dir = './record'
    if not os.path.isdir(record_dir):
        os.makedirs(record_dir)

    image_size = 224
    if params.method in ['jigsaw', 'imprint_jigsaw']:
        extra_data = 15     # extra_unlabeled data
        test_datamgr = SetDataManager(image_size, n_way=params.test_n_way, n_support=params.n_shot, n_query=params.n_query+extra_data, n_eposide=1)
        test_loader = test_datamgr.get_data_loader(test_file, aug=False)
        if params.dataset == "miniImagenet":
            num_class = 64
        elif params.dataset == "tieredImagenet":
            num_class = 351
        elif params.dataset == "caltech256":
            num_class = 257
        elif params.dataset == "CUB":
            num_class = 200  # set to 200 since the label range 0~199 even though there are only 100 classes to be trained
        else:
            raise ValueError('Unknown dataset')

        if params.method == "jigsaw":
            model = Jigsaw(num_class=num_class)
        else:
            model = ImprintJigsaw(num_class=num_class)
    else:
        raise ValueError('Unknown method')

    model = model.cuda()
    model.eval()

    checkpoint_dir = '%s/checkpoints/%s/ResNet18_%s' % (configs.save_dir, params.dataset, params.method)

    if params.save_iter != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(checkpoint_dir)
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    acc_all = []
    iter_num = 600
    for i in range(iter_num):
        print("Test case %d" % i)
        acc = model.test_loop(test_loader, params)
        acc_all.append(acc)

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

    with open('./record/results.txt', 'a') as f:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        exp_setting = '%s-ResNet18-%s %sshot %sway_test (replace %d(train)/[%d, %d](test) blocks)' % (params.dataset, params.method, params.n_shot, params.test_n_way, params.jig_replace_num_train, params.jig_replace_min_num, params.jig_replace_max_num)
        acc_str = '%d Test Acc = %4.2f%% +- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
        f.write( 'Time: %s, Setting: %s, Acc: %s \n' %(timestamp, exp_setting, acc_str))

