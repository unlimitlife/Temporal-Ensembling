import os
import pickle
import random
import copy
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data import IterableDataset
from PIL import Image
import bisect
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from .TinyImageNet import TinyImageNet
from .SubImageNet import SubImageNet
from .zca_bn import ZCA 

_datasets = {'cifar10': torchvision.datasets.CIFAR10,
             'cifar100': torchvision.datasets.CIFAR100,
             'mnist': torchvision.datasets.MNIST,
             'stl10': lambda data_path, train, download: torchvision.datasets.STL10(data_path,
                                                                                    split='train' if train else 'test',
                                                                                    download=download),
             'tiny_image': TinyImageNet,
             'sub_image': SubImageNet}

def zca_to_image(x):
    x = x.reshape((32,32,3))
    m,M = x.min(), x.max()
    x = (x - m) / (M - m)
    return Image.fromarray(np.uint8(x*255))

def preprocess(data_path, dataset, ZCA_=False):
    """ If the dataset does not exist, download it and create a dataset.
        Args:
            data_path (str): root directory of dataset.
            dataset (str): name of dataset.
    """
    il_data_path = os.path.join(data_path, 'zca_' + dataset)
    train_path = os.path.join(il_data_path, 'train')
    val_path = os.path.join(il_data_path, 'val')

    if os.path.isdir(il_data_path):
        return

    os.makedirs(train_path)
    os.makedirs(val_path)

    train_set = _datasets[dataset](data_path, train=True, download=True)
    val_set = _datasets[dataset](data_path, train=False, download=True)
    images = {}
    labels = {}
    if ZCA_ == True:
        for tag, cur_set, cur_path in [['train', train_set, train_path], ['test', val_set, val_path]]:
            for idx, item in enumerate(cur_set):
                images.setdefault(tag,[])
                images[tag].append(np.asarray(item[0],dtype='float32').reshape(-1,3,32,32) / np.float32(255))
                labels.setdefault(tag,[])
                labels[tag].append(np.asarray(item[1],dtype='int32'))
            images[tag] = np.concatenate(images[tag])
            labels[tag] = np.asarray(labels[tag])
        #import pdb; pdb.set_trace()
        whitener = ZCA(x=images['train'])
        #import sys; sys.exit()
        for tag, cur_path in [['train', train_path],['test', val_path]]:
            ###images[tag] = whitener.apply(images[tag])
            # Pad according to the amount of jitter we plan to have.
            for idx, (img, label) in enumerate(zip(images[tag], labels[tag])):
                img = zca_to_image(img)
                item = (img, label)
                if not os.path.exists(os.path.join(cur_path, str(label))):
                    os.makedirs(os.path.join(cur_path, str(label)))
                with open(os.path.join(cur_path, str(label), str(idx) + '.p'), 'wb') as f:
                    pickle.dump(item, f)
                
    # dump pickles for each class
    else:
        for cur_set, cur_path in [[train_set, train_path], [val_set, val_path]]:
            for idx, item in enumerate(cur_set):
                label = item[1]
                if not os.path.exists(os.path.join(cur_path, str(label))):
                    os.makedirs(os.path.join(cur_path, str(label)))
                with open(os.path.join(cur_path, str(label), str(idx) + '.p'), 'wb') as f:
                    pickle.dump(item, f)


class Taskset(data.Dataset):
    def __init__(self, root, train=True, transform=None, num_labels=4000, num_classes=10):
        """
        Args:
            root (str): root directory of dataset prepared for incremental learning (by preper_for_IL)
            task (list): list of classes that are assigned for the task
            task_idx (int): index of the task, ex) 2nd task among total 10 tasks
            train (bool): whether it is for train or not
            transform (callable) : transforms for dataset
            target_transform (callable) : transforms for target
        """
        if train:
            self.root = os.path.expanduser(root) + '/train'
        else:
            self.root = self.root = os.path.expanduser(root) + '/val'

        if not os.path.isdir(self.root):
            print('Exception: there is no such directory : {}'.format(self.root))

        self.train = train  # training set or test set
        self.num_labels = num_labels
        self.transform = transform

        self.targets = []
        self.filenames = []
        self.data = []
        self.un_data = []

        if self.train:
            for cls in os.listdir(self.root):
                chk = 0
                file_path = self.root + '/' + str(cls)
                files = os.listdir(file_path)
                random.shuffle(files)
                #for file in os.listdir(file_path):
                for file in files:
                    with open(file_path + '/' + file, 'rb') as f:
                        if chk < int(self.num_labels/num_classes):
                            entry = pickle.load(f)
                            self.data.append(entry[0])
                            self.targets.append(entry[1])
                            self.filenames.append(file)
                        else :
                            entry = pickle.load(f)
                            self.un_data.append(entry[0])
                            self.filenames.append(file)
                        chk += 1
        else: 
            for cls in os.listdir(self.root):
                file_path = self.root + '/' + str(cls)
                for file in os.listdir(file_path):
                    with open(file_path + '/' + file, 'rb') as f:
                        entry = pickle.load(f)
                        self.data.append(entry[0])
                        self.targets.append(entry[1])
                        self.filenames.append(file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, soft_label) where target is index of the target class.
        """
        idx = index

        if self.train:
            if idx < self.num_labels:
                return self.transform(self.data[idx]), self.targets[idx], idx
            else:
                return self.transform(self.un_data[idx-self.num_labels]), -1, idx
        else:
            img, target = self.data[idx], int(self.targets[idx])
            img = self.transform(img)

            return img, target, idx

    def __len__(self):
        return len(self.data) + len(self.un_data)


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())
    from config import config

    for dataset in _datasets:
        preprocess(config['data_path'], dataset)
