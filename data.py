import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tifffile
import glob
import os
import pandas as pd
from torchvision.io import read_image



preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

#order of classes for dataloader ['sparseresidential', 'mediumresidential', 'denseresidential']
## to fix : currently converting to PIL then tensor, is there another way?
class UCmerced(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        if self.transform==None:
            self.transform = preprocess
        self.target_transform = target_transform
        self.classes = ['sparseresidential', 'mediumresidential', 'denseresidential']
        self.clss_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples= []
        self.domain_label = 1
        for cls_idx,cls_name in enumerate(self.classes):
            cls_dir = os.path.join(img_dir,cls_name)
            for sample_name in os.listdir(cls_dir):
                sample_path = os.path.join(cls_dir,sample_name)
                self.samples.append((sample_path,cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path,cls_idx = self.samples[idx]
        image = tifffile.imread(sample_path)
        domain_label = self.domain_label
        label = cls_idx
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, domain_label

class NAIP(Dataset):
    def __init__(self,img_dir,transform = None, target_transform = None):
        self.img_dir = img_dir
        self.transform = transform
        if self.transform == None:
            self.transform = preprocess
        self.target_transform = target_transform
        self.classes = ['Sparse', 'Medium', 'Dense']
        self.clss_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples= []
        self.domain_label = 0
        for cls_idx,cls_name in enumerate(self.classes):
            cls_dir = os.path.join(img_dir,cls_name)
            for sample_name in os.listdir(cls_dir):
                sample_path = os.path.join(cls_dir,sample_name)
                self.samples.append((sample_path,cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path,cls_idx = self.samples[idx]
        image = read_image(sample_path)
        domain_label = self.domain_label
        label = cls_idx
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, domain_label

def split_dataset(dataset, validation_fraction = 0.2):
    size = len(dataset)
    val_size = int(0.2*size)
    train_size = size - val_size
    train,val = random_split(dataset,(train_size,val_size))
    return train,val
