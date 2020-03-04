"""
Continual Local Replacement with weight imprinting
"""

import utils
import time

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import tile

patch_xl = np.array([0,0,0,74,74,74,148,148,148])
patch_xr = np.array([74,74,74,148,148,148,224,224,224])
patch_yl = np.array([0,74,148,0,74,148,0,74,148])
patch_yr = np.array([74,148,224,74,148,224,74,148,224])

class LinearClassifier(nn.Module):
    def __init__(self, feature_dim, num_class):
        super(LinearClassifier, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([10]))
        self.fc = nn.Linear(feature_dim, num_class, bias=False)

    def forward(self, x):
        out = self.scale * x
        logits = self.fc(out)
        return logits

class ImprintJigsaw(nn.Module):
    """ partial codes are referred from https://github.com/YU1ut/imprinted-weights"""
    def __init__(self, num_class):
        super(ImprintJigsaw, self).__init__()
        self.num_class = num_class

        self.feature = torchvision.models.resnet18(pretrained=False)
        self.feature_dim = self.feature.fc.in_features
        self.feature.fc = utils.Identity()  # discard last fc layer

        self.scale = nn.Parameter(torch.FloatTensor([10]))
        self.classifier = nn.Linear(self.feature_dim, num_class, bias=False)

        self.ceLoss = nn.CrossEntropyLoss()

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)
        return output

    def forward_before_classifier(self, x):
        out = self.feature.forward(x)
        out = self.l2_norm(out)
        return out

    def forward(self, x):
        out = self.feature.forward(x)
        out = self.l2_norm(out)
        out = self.scale * out
        logits = self.classifier.forward(out)
        return logits

    def forward_loss(self, x, y):
        logits = self.forward(x)
        return self.ceLoss(logits, y)

    def train_loop(self, epoch, train_loader, base_jigsaw_loader, optimizer):
        print_freq = 10
        avg_loss = 0

        i = 0
        for (x, y), (x_jigsaw, y_jigsaw) in zip(train_loader, base_jigsaw_loader):
            x = x.cuda()
            y = y.long().cuda()
            x_jigsaw = x_jigsaw.cuda()
            y_jigsaw = y_jigsaw.long().cuda()

            loss1 = self.forward_loss(x, y)
            loss2 = self.forward_loss(x_jigsaw, y_jigsaw)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # normalize classifier weight
            w = self.classifier.weight.data
            norm = w.norm(p=2, dim=1, keepdim=True)
            self.classifier.weight.data = w.div(norm.expand_as(w))

            avg_loss = avg_loss + loss.item()

            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss / float(i + 1)))
            i+=1

    def test_loop(self, test_loader, params):
        acc_all = []

        self.params = params
        self.jig_replace_min_num = params.jig_replace_min_num
        self.jig_replace_max_num = params.jig_replace_max_num

        self.n_query = params.n_query
        self.n_way = params.test_n_way
        self.n_support = params.n_shot
        self.repeat_time = 1

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            x = x.cuda()
            self.n_unlabeled = x.size(1) - self.n_support - self.n_query

            start = time.time()
            correct_this, count_this = self.adapt_with_jigsaw(x)
            end = time.time()
            print("consume time "+str(end-start)+" s")

            acc_all.append(correct_this / count_this * 100)

        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))

        return acc_mean

    def adapt_with_jigsaw(self, x):
        """ backbone is fixed """
        # input data
        if self.n_support == 1:
            self.repeat_time = 5  # repeat support data if shot =1
        x_support = x[:, :self.n_support]
        x_support = tile(x_support, dim=1, n_tile=self.repeat_time)
        x_support = x_support.contiguous().view(self.n_way * self.n_support * self.repeat_time, *x.size()[2:])
        y_support = torch.from_numpy(np.repeat(range(self.n_way), self.n_support * self.repeat_time)).long().cuda()

        x_query = x[:, self.n_support:(self.n_support+self.n_query)]
        x_query = x_query.contiguous().view(self.n_way * self.n_query, *x.size()[2:])

        x_unlabeled = x[:, (self.n_support+self.n_query):]
        x_unlabeled = x_unlabeled.contiguous().view(self.n_way * self.n_unlabeled, *x.size()[2:])

        self.eval()
        with torch.no_grad():
            z_support = self.feature(x_support)
            imprinting_weights = self.l2_norm(z_support)

            # calculate feature before hand since backbone won't be updated
            z_query = self.forward_before_classifier(x_query)
            z_unlabeled = self.forward_before_classifier(x_unlabeled)

        # imprinted weight
        weights = torch.zeros(self.n_way, self.feature_dim)
        for i in range(self.n_way):
            tmp = imprinting_weights[y_support == i].mean(0)
            weights[i] = tmp / tmp.norm(p=2)

        # classifier + opt
        classifier = LinearClassifier(self.feature_dim, self.n_way)
        classifier.fc.weight.data = weights        # inject imprinted weights
        classifier = classifier.cuda()
        optimizer = torch.optim.Adam(classifier.parameters())

        batch_size = 4
        support_size = y_support.size(0)

        x_support_aug = x_support
        for epoch in range(100):
            classifier.train()
            rand_id = np.random.permutation(support_size)
            # rand_id = np.array([i for i in range(support_size)])
            for i in range(0, support_size, batch_size):
                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).long().cuda()
                x_batch = x_support_aug[selected_id]
                y_batch = y_support[selected_id]

                with torch.no_grad():
                    z = self.forward_before_classifier(x_batch)     # backbone is fixed

                logits = classifier(z)
                loss = self.ceLoss(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # normalize classifier weight
                w = classifier.fc.weight.data
                norm = w.norm(p=2, dim=1, keepdim=True)
                classifier.fc.weight.data = w.div(norm.expand_as(w))

            if epoch < 99:  # no aug in last epoch
                # different data selection strategies for ablation study
                label2idx, topk_labels = self.selected_by_pseudo_labels(classifier, z_unlabeled)

                selected_idx = torch.cat(label2idx, dim=0)
                selected_unlabel_img = x_unlabeled[selected_idx]

                # jigsaw aug
                x_support_aug = self.jigsaw_aug(x_support, selected_unlabel_img, min_replace_block_num=self.jig_replace_min_num, max_replace_block_num=self.jig_replace_max_num)
                ####################

        # test on query set after adaptation
        scores = classifier(z_query)

        """Get the accuracy"""
        # query labels
        y_query = np.repeat(range(self.n_way), self.n_query)

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        correct = np.sum(topk_ind[:, 0] == y_query)
        count = len(y_query)

        return correct, count

    ##############################################################
    # The strategies to pick up unlabeled data. For ablation study
    ##############################################################
    def selected_by_random(self):
        # randomly generate pseudo labels
        random_y_unlabeled = torch.LongTensor(self.n_way*self.n_unlabeled).random_(0, self.n_way).cuda()

        init_idx = torch.randperm(self.n_way*self.n_unlabeled)
        label2idx = []
        for i in range(self.n_way):
            label2idx.append(init_idx[i * self.n_support:(i + 1) * self.n_support])

        topk_labels = random_y_unlabeled
        return label2idx, topk_labels

    def selected_by_pseudo_labels(self, model, unlabeled):
        model.eval()
        with torch.no_grad():
            unlabeled_logits = model.forward(unlabeled)  # pseudo label
        topk_scores, topk_labels = unlabeled_logits.data.topk(1, 1, True, True)
        topk_scores = topk_scores.view(-1)
        topk_labels = topk_labels.view(-1)

        init_idx = torch.randperm(unlabeled.size(0))
        label2idx = []

        n_support = self.n_support *self.repeat_time
        for i in range(self.n_way):
            label2idx.append(init_idx[i * n_support:(i + 1) * n_support])
            idx = torch.nonzero(topk_labels == i).view(-1)
            # shuffle pseudo labels (randomly select from corresponding class)
            idx = idx[torch.randperm(idx.size(0))]
            k = min(idx.size(0), n_support)
            label2idx[i][:k] = idx[:k]

        return label2idx, topk_labels

    def selected_by_gt(self):
        y_unlabeled = torch.from_numpy(np.repeat(range(self.n_way), self.n_unlabeled)).long().cuda()
        label2idx = []
        for i in range(self.n_way):
            idx = torch.nonzero(y_unlabeled == i).view(-1)
            # shuffle pseudo labels
            idx = idx[torch.randperm(idx.size(0))]
            k = min(idx.size(0), self.n_support)
            label2idx.append(idx[:k])

        topk_labels = y_unlabeled
        return label2idx, topk_labels


    ####################################
    # jigsaw augmentations
    ####################################
    def jigsaw_aug(self, images, auxiliary_images, min_replace_block_num=0, max_replace_block_num=2):
        replace_block_num = np.random.randint(min_replace_block_num, max_replace_block_num + 1)
        replaced_indexs = np.random.choice(9, replace_block_num, replace=False)

        aug_images = images.clone()
        for l in range(replace_block_num):
            replaced_index = int(replaced_indexs[l])
            aug_images[:,:, patch_xl[replaced_index]:patch_xr[replaced_index], patch_yl[replaced_index]:patch_yr[replaced_index]] = auxiliary_images[:,:,
                                                                                    patch_xl[replaced_index]:patch_xr[replaced_index],
                                                                                    patch_yl[replaced_index]:patch_yr[replaced_index]]

        return aug_images



