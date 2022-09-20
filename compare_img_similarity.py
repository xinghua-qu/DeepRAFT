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
from image_utils import FGSM_DEC, PGD_DEC
import matplotlib.pyplot as plt 
import csv
import numpy as np
import os
import pandas as pd
import torchvision.transforms as T
from autoattack import AutoAttack
from DiffJPEG.DiffJPEG import DiffJPEG
import math

class PSNR(object):
    def __init__(self, des="Peak Signal to Noise Ratio"):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim=1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1 / mse)

class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''
    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        max_val = 1
        min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret


class LPIPS(object):
    '''
    borrowed from https://github.com/richzhang/PerceptualSimilarity
    '''
    def __init__(self, des="Learned Perceptual Image Patch Similarity", version="0.1"):
        self.des = des
        self.version = version
        self.model = lpips.LPIPS(net="alex", verbose=False).cuda()

    def __repr__(self):
        return "LPIPS"

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        return self.model(y_pred, y_true)

def calculate_similarity(encoder, detector, test_loader, args):
    psnr_list = []
    ssim_list = []
    lps_list  = []
    for batch_idx, (image_input) in enumerate(test_loader, 0):
        image_input = image_input.cuda() # not necessary since it is already on GPU 
        stamp_perturbation = encoder(image_input )
        stamped_image = image_input + stamp_perturbation
        psnr    = PSNR()
        ssim    = SSIM()
        lps     = LPIPS()
        val_psnr = psnr(stamped_image, image_input)
        val_ssim = ssim(stamped_image, image_input)
        val_lps  = lps(stamped_image.cuda(), image_input.cuda())
        psnr_list.append(val_psnr.item())
        ssim_list.append(val_ssim.item())
        lps_list.append(val_lps.squeeze().mean().item())

    print(np.mean(psnr_list), np.std(psnr_list) )
    print(np.mean(ssim_list), np.std(ssim_list)   )
    print(np.mean(lps_list), np.std(lps_list) )

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
    datasets = train_test_dataset(dataset, test_split=0.2)
    test_dataset = datasets['test']
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    calculate_similarity(encoder, detector, test_loader, args)