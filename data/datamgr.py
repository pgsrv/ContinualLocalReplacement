# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import sys
import json
import random
import torch
from PIL import Image
import numpy as np

from data.transforms import *
from data.dataset import SimpleDataset, JigsawDataset, SetDataset, EpisodicBatchSampler
from abc import abstractmethod


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(data_file, transform)
        if sys.platform == "win32":     # Note: windows system doesn't support num_workers multiple threads
            data_loader_params = dict(batch_size=self.batch_size, shuffle=True, pin_memory=True)
        elif sys.platform == "linux":
            data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        else:
            assert False, "Unknown OS!"
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way, n_support, n_query, n_eposide =100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset( data_file , self.batch_size, transform )
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
        if sys.platform == "win32":     # Note: windows system doesn't support num_workers multiple threads
            data_loader_params = dict(batch_sampler=sampler, pin_memory=True)
        elif sys.platform == "linux":
            data_loader_params = dict(batch_sampler=sampler, num_workers=8, pin_memory=True)
        else:
            assert False, "Unknown OS!"
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

#######################################
# Jigsaw dataset
#######################################
class JigsawDataManger(DataManager):
    def __init__(self, image_size, batch_size, max_replace_block_num=4):
        super(JigsawDataManger, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.max_replace_block_num = max_replace_block_num

    def get_data_loader(self, data_file, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = JigsawDataset(data_file, transform, max_replace_block_num=self.max_replace_block_num)
        if sys.platform == "win32":     # Note: windows system doesn't support num_workers multiple threads
            data_loader_params = dict(batch_size=self.batch_size, shuffle=True, pin_memory=True)
        elif sys.platform == "linux":
            data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        else:
            assert False, "Unknown OS!"
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

