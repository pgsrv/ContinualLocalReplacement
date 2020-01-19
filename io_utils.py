import numpy as np
import os
import glob
import argparse

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--gpu', default='0', help='gpu id to run')
    parser.add_argument('--dataset'     , default='miniImagenet', help='CUB/miniImagenet/caltech256/tieredImagenet')
    parser.add_argument('--method'      , default='jigsaw',   help='jigsaw/imprint_jigsaw')
    parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
    parser.add_argument('--n_query', default=15, type=int, help='number of data for training or testing')

    # params for jigsaw methods
    parser.add_argument('--jig_replace_num_train', default=4, type=int, help='the number of replaced blocks in jigsaw augmentation during training')
    parser.add_argument('--jig_replace_min_num', default=0, type=int, help='the minimum number of replaced blocks in jigsaw augmentation during testing')
    parser.add_argument('--jig_replace_max_num', default=6, type=int, help='the maximum number of replaced blocks in jigsaw augmentation during testing')   # A larger number e.g. 9 may achieve better performance on CUB dataset especially for imprinting_jigsaw

    if script == 'train':
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=-1, type=int, help ='Stopping epoch')
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
    elif script == 'test':
        parser.add_argument('--save_iter', default=-1, type=int,help ='saved weights from the model trained in x epoch, use the best model if x is -1')
    else:
       raise ValueError('Unknown script')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
