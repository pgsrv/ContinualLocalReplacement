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
from io_utils import parse_args, get_resume_file


def train(base_loader, base_jigsaw_loader, val_loader, model, start_epoch, stop_epoch, params):
    max_acc = 0
    acc = -1
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, base_jigsaw_loader, optimizer)

        if params.dataset in ['miniImagenet', 'tieredImagenet']:
            if epoch % 4 == 0 or epoch == stop_epoch - 1:   # evaluate every 4 epochs to speed training procedure
                model.eval()
                acc = model.test_loop(val_loader, params)
        else:
            model.eval()
            acc = model.test_loop(val_loader, params)

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if acc > max_acc:
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    params = parse_args('train')

    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'

    if params.method in ['jigsaw', 'imprint_jigsaw']:
        if params.dataset in ['miniImagenet']:
            params.stop_epoch = 400
            batch_size = 128
        elif params.dataset in ['caltech256', 'CUB']:
            params.stop_epoch = 100
            batch_size = 64
        elif params.dataset in ['tieredImagenet']:
            params.stop_epoch = 100
            batch_size = 256
        else:
            params.stop_epoch = 400  # default
            batch_size = 128
    else:  # meta-learning methods
        raise ValueError('Unknown methods')

    image_size = 224
    base_datamgr = SimpleDataManager(image_size, batch_size=batch_size)
    base_loader = base_datamgr.get_data_loader(base_file, aug=True)

    base_jigsaw_datamgr = JigsawDataManger(image_size, batch_size=batch_size, max_replace_block_num=params.jig_replace_num_train)
    base_jigsaw_loader = base_jigsaw_datamgr.get_data_loader(base_file, aug=False)

    extra_data = 15     # extra_unlabeled data
    val_datamgr = SetDataManager(image_size, n_way=params.test_n_way, n_support=params.n_shot, n_query=params.n_query+extra_data, n_eposide=50)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    if params.dataset == "miniImagenet":
        num_class = 64
    elif params.dataset == "tieredImagenet":
        num_class = 351
    elif params.dataset == "caltech256":
        num_class = 257
    elif params.dataset == "CUB":
        num_class = 200     # set to 200 since the label range 0~199 even though there are only 100 classes to be trained
    else:
        raise ValueError('Unknown dataset')

    if params.method == "jigsaw":
        model = Jigsaw(num_class=num_class)
    else:
        model = ImprintJigsaw(num_class=num_class)

    model = model.cuda()

    params.checkpoint_dir = '%s/checkpoints/%s/ResNet18_%s' % (configs.save_dir, params.dataset, params.method)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.resume:
        resume_file = get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])

    model = train(base_loader, base_jigsaw_loader, val_loader, model, start_epoch, stop_epoch, params)
