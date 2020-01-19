import os, glob
import shutil
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import skimage.transform

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from data.datamgr import UnNormalize
import torchvision.transforms as transforms

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1).long(), 1)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x