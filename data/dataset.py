# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os

# jigsaw patch positions
patch_xl = np.array([0,0,0,74,74,74,148,148,148])
patch_xr = np.array([74,74,74,148,148,148,224,224,224])
patch_yl = np.array([0,74,148,0,74,148,0,74,148])
patch_yr = np.array([74,148,224,74,148,224,74,148,224])

identity = lambda x:x
class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class JigsawDataset:
    def __init__(self, data_file, transform, max_replace_block_num=4, target_transform=identity):
        self.max_replace_block_num = max_replace_block_num
        self.transform = transform
        self.target_transform = target_transform

        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x, y in zip(self.meta['image_names'], self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.meta['image_labels'] = np.array(self.meta['image_labels'])

        self.original_size = len(self.meta['image_names'])

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])

        # ori_im = img.clone()

        if self.max_replace_block_num == 0:
            replace_block_num = 0
            replaced_indexs = []
        else:
            replace_block_num = np.random.randint(1, self.max_replace_block_num+1)
            replaced_indexs = np.random.choice(9, replace_block_num, replace=False)

        is_same_cls = np.random.randint(0, 2)

        if is_same_cls == 0:  # use a random image
            choose = np.random.randint(0, self.original_size)
            auxiliary_image_path = os.path.join(self.meta['image_names'][choose])
            auxiliary_image = Image.open(auxiliary_image_path).convert('RGB')
            auxiliary_image = self.transform(auxiliary_image)
        else:                 # use an image in same class
            labels = self.meta['image_labels']
            same_cls_idxs = np.where(labels == target)[0]
            choose = np.random.choice(same_cls_idxs, 1)[0]
            auxiliary_image_path = os.path.join(self.meta['image_names'][choose])
            auxiliary_image = Image.open(auxiliary_image_path).convert('RGB')
            auxiliary_image = self.transform(auxiliary_image)

        for l in range(replace_block_num):
            replaced_index = replaced_indexs[l]
            img[0:3, patch_xl[replaced_index]:patch_xr[replaced_index], patch_yl[replaced_index]:patch_yr[replaced_index]] = auxiliary_image[0:3,
                                                                                    patch_xl[replaced_index]:patch_xr[replaced_index],
                                                                                    patch_yl[replaced_index]:patch_yr[replaced_index]]

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, shuffle=True):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        self.shuffle = shuffle

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            if self.shuffle:
                extracted_cls = torch.randperm(self.n_classes)[:self.n_way]
            else:
                extracted_cls = [x for x in range(self.n_classes)][:self.n_way]
            yield extracted_cls




