import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image, ImageOps
from torch import optim

import utils
from dataset import StegaData
from dataset import train_test_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.tensorboard import SummaryWriter
import lpips
import argparse
from kornia import color
from torch.nn import functional as F
import torchvision
from utils_manupulator import FGSM_DEC, PGD_DEC, Traditional_Img_Manupulator
import matplotlib.pyplot as plt 
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as T
from autoattack import AutoAttack
from DiffJPEG.DiffJPEG import DiffJPEG
from evaluate import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=2000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--l2_loss_scale', type=float, default=2)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--rs_w', type=float, default=0.1)
    parser.add_argument('--rtt_w', type=float, default=1)
    parser.add_argument('--crp_w', type=float, default=1)
    parser.add_argument('--crp_dropout', type=float, default=0.9)
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--secret_size', type=int, default=50)
    user_args = parser.parse_args()
    step = int(78000)
    encoder_path = "results/{}/saved_models/encoder.pth".format(user_args.run_name)
    detector_path = "results/{}/saved_models/detector.pth".format(user_args.run_name)
    encoder = torch.load(encoder_path)
    detector = torch.load(detector_path)
    with open('setting.yaml', 'r') as f:
        fixed_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    args = EasyDict(dict( vars(user_args), **fixed_args ))
    dataset = StegaData(args.train_path, size=(400, 400))
    datasets = train_test_dataset(dataset, test_split=0.01)
    test_dataset = datasets['test']
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    evaluate(encoder, detector, test_loader, args)