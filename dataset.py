import os
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from image_utils import Image_Operations
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


class StegaData(Dataset):
    def __init__(self, data_path, size=(400, 400)):
        self.data_path = data_path
        self.size = size
        self.files_list  = glob(os.path.join(self.data_path, '*.jpg'))
        self.to_tensor   = transforms.ToTensor()
        self.IM_operator = Image_Operations()

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]

        img_cover = Image.open(img_cover_path).convert('RGB')
        img_cover = ImageOps.fit(img_cover, self.size)
        img_cover = self.to_tensor(img_cover) # the value has been scaled to [0,1]

        return img_cover

    def __len__(self):
        return len(self.files_list)


def train_test_dataset(dataset, test_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, val_idx)
    return datasets
