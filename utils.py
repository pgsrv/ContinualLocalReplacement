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

#####################################################################
# activation map
#####################################################################
class ExtractFeatureMap():
    features=None
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.detach()

    def remove(self):
        self.hook.remove()

def activation_map(classifier, x_support, fm_support, z_support, x_unlabeled, fm_unlabeled, z_unlabeled, x_support_aug, fm_support_aug, z_support_aug):
    """Visualization for interpretable"""
    batch_size, d, h, w = fm_support.shape

    fc_w = classifier.weight
    with torch.no_grad():
        # get predicted label
        logit_support = classifier(z_support)
        logit_unlabeled = classifier(z_unlabeled)
        logit_support_aug = classifier(z_support_aug)

        _, topk_labels_support = logit_support.topk(1, 1, True, True)
        topk_labels_support = topk_labels_support.view(-1)

        _, topk_labels_unlabeled = logit_unlabeled.topk(1, 1, True, True)
        topk_labels_unlabeled = topk_labels_unlabeled.view(-1)

        _, topk_labels_support_aug = logit_support_aug.topk(1, 1, True, True)
        topk_labels_support_aug = topk_labels_support_aug.view(-1)

        # activation on support set
        support_fc_w = fc_w[topk_labels_support].unsqueeze(2)
        fm_support = fm_support.view(batch_size, d, h*w).transpose(1, 2)
        act_map_support = torch.bmm(fm_support, support_fc_w)
        act_map_support = act_map_support.view(batch_size, h, w)
        for i in range(batch_size):         # normalize to [0, 1], TODO: it may be implemented by more efficient way
            act_map_support[i] -= torch.min(act_map_support[i])
            act_map_support[i] /= torch.max(act_map_support[i])

        # activation on unlabeled set
        unlabeled_fc_w = fc_w[topk_labels_unlabeled].unsqueeze(2)
        fm_unlabeled = fm_unlabeled.view(batch_size, d, h*w).transpose(1, 2)
        act_map_unlabeled = torch.bmm(fm_unlabeled, unlabeled_fc_w)
        act_map_unlabeled = act_map_unlabeled.view(batch_size, h, w)
        for i in range(batch_size):         # normalize to [0, 1]
            act_map_unlabeled[i] -= torch.min(act_map_unlabeled[i])
            act_map_unlabeled[i] /= torch.max(act_map_unlabeled[i])

        # activation on augmented set
        support_aug_fc_w = fc_w[topk_labels_support_aug].unsqueeze(2)
        fm_support_aug = fm_support_aug.view(batch_size, d, h * w).transpose(1, 2)
        act_map_support_aug = torch.bmm(fm_support_aug, support_aug_fc_w)
        act_map_support_aug = act_map_support_aug.view(batch_size, h, w)

        for i in range(batch_size):  # normalize to [0, 1]
            act_map_support_aug[i] -= torch.min(act_map_support_aug[i])
            act_map_support_aug[i] /= torch.max(act_map_support_aug[i])

        # visualize
        show_index = [0, 10, 20]
        act_map_support = act_map_support.cpu().numpy()
        act_map_unlabeled = act_map_unlabeled.cpu().numpy()
        act_map_support_aug = act_map_support_aug.cpu().numpy()

        nrow = 2
        ncol = 3

        # for i in range(batch_size):
        #     if i >= 3:
        #         break
        for i in show_index:
            act_map_sup = skimage.transform.resize(act_map_support[i], x_support.shape[2:], mode='constant')
            support_image = UnNormalize().get_unnormalize_transform()(x_support[i]).cpu().numpy()
            support_image = np.dstack((support_image[0], support_image[1], support_image[2]))

            act_map_un = skimage.transform.resize(act_map_unlabeled[i], x_unlabeled.shape[2:], mode='constant')
            unlabeled_image = UnNormalize().get_unnormalize_transform()(x_unlabeled[i]).cpu().numpy()
            unlabeled_image = np.dstack((unlabeled_image[0], unlabeled_image[1], unlabeled_image[2]))

            act_map_sup_aug = skimage.transform.resize(act_map_support_aug[i], x_support_aug.shape[2:], mode='constant')
            support_aug_image = UnNormalize().get_unnormalize_transform()(x_support_aug[i]).cpu().numpy()
            support_aug_image = np.dstack((support_aug_image[0], support_aug_image[1], support_aug_image[2]))

            # no gap between images
            # https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
            fig = plt.figure(figsize=((ncol + 1)*2, (nrow + 1)*2))
            gs = gridspec.GridSpec(nrow, ncol,
                                   wspace=0.0, hspace=0.0,
                                   top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                                   left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

            sub_figs0 = plt.subplot(gs[0])
            sub_figs0.imshow(support_image)
            sub_figs1 = plt.subplot(gs[1])
            sub_figs1.imshow(unlabeled_image)
            sub_figs2 = plt.subplot(gs[2])
            sub_figs2.imshow(support_aug_image)
            sub_figs3 = plt.subplot(gs[3])
            sub_figs3.imshow(support_image)
            sub_figs3.imshow(act_map_sup, alpha=0.5, cmap='jet')
            sub_figs4 = plt.subplot(gs[4])
            sub_figs4.imshow(unlabeled_image)
            sub_figs4.imshow(act_map_un, alpha=0.5, cmap='jet')
            sub_figs5 = plt.subplot(gs[5])
            sub_figs5.imshow(support_aug_image)
            sub_figs5.imshow(act_map_sup_aug, alpha=0.5, cmap='jet')

            for ax in fig.axes:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
        print()


#####################################################################
# Save training and testing data to observe the relationship with ACC
#####################################################################
class DataOB:
    def __init__(self, method, save_root="./record/train_test_data", max_storage_in_each_interval=10):
        self.method = method
        self.save_root = save_root
        self.max_storage_in_each_interval = max_storage_in_each_interval
        self.first_saving = True
        self.saved_count = dict()

    def set_acc_intervals(self, dataset_and_setting):
        """Manually set ACC intervals since ACC varies from different datasets and settings"""
        # miniImagenet 5-way 5-shot
        if dataset_and_setting == "miniImagenet_5_shot_5_way":
            acc_keys = {"below30": [0, 30], "30-40": [30, 40], "40-50": [40, 50], "50-60": [50, 60], "60-70": [60, 70], "70-80": [70, 80], "80-90": [80, 90], "above90": [90, 101]}    # the interval is [a, b)
        # miniImagenet 5-way 1-shot
        elif dataset_and_setting == "miniImagenet_1_shot_5_way":
            acc_keys = {"below30": [0, 30], "30-40": [30, 40], "40-50": [40, 50], "50-60": [50, 60], "60-70": [60, 70], "70-80": [70, 80], "above80": [80, 101]}
        # TODO: Maybe more settings
        else:
            raise ValueError("Unknown dataset and setting")
        return acc_keys

    def save_tensor2img(self, x, acc, acc_per_cls, n_support, n_query, dataset_and_setting, same_episode=False):
        """Save the input with different ACC"""
        if same_episode:
            save_root = os.path.join(self.save_root, self.method + "_" + dataset_and_setting+"_same_episode_different_support")
        else:
            save_root = os.path.join(self.save_root, self.method+"_"+dataset_and_setting)

        if self.first_saving:
            self.first_saving = False
            if os.path.exists(save_root):
                shutil.rmtree(save_root)

        acc_keys = self.set_acc_intervals(dataset_and_setting)

        # find the corresponding interval and its acc_key
        acc_key = "unknown"
        for k, v in acc_keys.items():
            low, high = v
            if acc>=low and acc<high:
                acc_key = k
                break
        if acc_key == "unknown":
            raise ValueError("Cannot find correct acc_key")

        # control max number of data in each interval
        is_save = True
        if acc_key in self.saved_count and self.saved_count[acc_key] >= self.max_storage_in_each_interval:
            is_save = False

        if is_save:
            if acc_key in self.saved_count:
                self.saved_count[acc_key] += 1
            else:
                self.saved_count[acc_key] = 1

            save_dir = os.path.join(save_root, acc_key, "data%d_acc_%.3f"%(self.saved_count[acc_key], acc))

            for cls in range(x.size(0)):
                cls_acc = acc_per_cls[cls]
                im_save_dir = os.path.join(save_dir, "class%d_acc%.3f"%(cls, cls_acc))
                os.makedirs(im_save_dir, exist_ok=True)

                for n in range(0, n_support+n_query):       # only save support and query set. Discard unlabeled images
                    im = x[cls][n]
                    im = UnNormalize().get_unnormalize_transform()(im).cpu()
                    image = transforms.ToPILImage()(im)
                    # im = np.dstack((im[0], im[1], im[2]))
                    # im = np.uint8(im)
                    # image = Image.fromarray(im)
                    # image.show()
                    image.save(os.path.join(im_save_dir, "%d.jpg"%n))