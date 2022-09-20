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
from torch.utils.tensorboard import SummaryWriter
import lpips
import argparse
from kornia import color
from torch.nn import functional as F
import torchvision
from evaluate import evaluate
from utils_manupulator import FGSM_DEC, PGD_DEC, Traditional_Img_Manupulator
import matplotlib.pyplot as plt 
from evaluate import evaluate
from image_utils import PGD_DEC, Image_Operations

def main(user_args):
    
    # get the argument
    with open('setting.yaml', 'r') as f:
        fixed_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    args = EasyDict(dict( user_args, **fixed_args ))
    
    dataset_dir = 'mirflickr'
    cc_data_dir = './cc_{}/'.format(dataset_dir)
    rtt_data_dir = './rtt_{}/'.format(dataset_dir)

    IM_operator = Image_Operations()
    dataset = StegaData(args.train_path, args.secret_size, size=(400, 400))
    
    for epoch in range(args.max_epoch):      
        for batch_idx, (image_input, secret_input) in enumerate(train_loader, 0):
            image_input = image_input.cuda() # not necessary since it is already on GPU 
            secret_input = secret_input.cuda()
            stamp_perturbation = encoder(image_input, secret_input)
            stamped_image = image_input + stamp_perturbation
            
            predicted_ones  = detector(image_input)
            predicted_zeros = detector(stamped_image)
            non_stamped_det_loss = F.binary_cross_entropy(predicted_ones, ones)
            stamped_det_loss = F.binary_cross_entropy(predicted_zeros, zeros)
            
            ## rotate
            # degrees = [-20, 20]
            # rtt_img_input = IM_operator.rotate(image_input, degrees)
            # rtt_stmp_img  = IM_operator.rotate(stamped_image, degrees)
            # rtt_ones  = detector(rtt_img_input)
            # rtt_zeros = detector(rtt_stmp_img)
            # loss_rtt = F.binary_cross_entropy(rtt_ones, ones)+F.binary_cross_entropy(rtt_zeros, zeros)
            
            # ## crop
            # cc_sizes = [200, 400]
            # crp_img_input = IM_operator.center_crop(image_input, cc_sizes)
            # crp_stmp_img  = IM_operator.center_crop(stamped_image, cc_sizes)
            # crp_ones  = detector(crp_img_input)
            # crp_zeros = detector(crp_stmp_img)
            # loss_crp = F.binary_cross_entropy(crp_ones, ones)+F.binary_cross_entropy(crp_zeros, zeros)
            
            ## Randomized Smoothing
            # augment inputs with noise
            rs_std = args.std
            ptb_image_input = image_input + torch.randn_like(image_input, device='cuda') * rs_std
            ptb_image_input = torch.clamp(ptb_image_input,-1,1)
            ptb_stamped_image = stamped_image + torch.randn_like(stamped_image, device='cuda') * rs_std
            ptb_stamped_image = torch.clamp(ptb_stamped_image,-1,1)
            rs_ones  = detector(ptb_image_input)
            rs_zeros = detector(ptb_stamped_image)
            loss_rs = F.binary_cross_entropy(rs_ones, ones)+F.binary_cross_entropy(rs_zeros, zeros)
            
            normalized_input = image_input * 2 - 1
            normalized_encoded = stamped_image * 2 - 1
            lpips_loss = torch.mean(lpips_alex(normalized_input, normalized_encoded)) # imperceptibility of stamp perturbation
            
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            encoded_image_yuv = color.rgb_to_yuv(stamped_image)
            image_input_yuv = color.rgb_to_yuv(image_input)
            im_diff = encoded_image_yuv - image_input_yuv
            yuv_loss = torch.mean((im_diff) ** 2, axis=[0, 2, 3])
            yuv_scales = torch.Tensor(yuv_scales)
            if args.cuda:
                yuv_scales = yuv_scales.cuda()
            image_loss = torch.dot(yuv_loss, yuv_scales) # L2 norm for image similarity loss
            
            if predicted_ones.size()!=ones.size():
                print(image_input.size(), secret_input.size(), predicted_ones.size(), ones.size())
            
            loss_all = image_loss + lpips_loss + non_stamped_det_loss + stamped_det_loss + args.rs_w*loss_rs
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            if global_step % args.log_interval == 0:
                name_space = ['image_loss:' , 'lpips_loss:' , 'non_stamp_loss:' , 'stamp_loss:',  'loss_rs:']
                losses_log = [image_loss.item() , lpips_loss.item() , non_stamped_det_loss.item() , stamped_det_loss.item(),  loss_rs.item()]
                print('Global_step {}: {}'.format(global_step, ['{}{}'.format(name_space[i],losses_log[i]) for i in range(len(losses_log))]))
            if global_step % args.save_interval == 0:
                torch.save(encoder, os.path.join(args.saved_models, "encoder_{}.pth".format(global_step)))
                torch.save(detector, os.path.join(args.saved_models, "detector_{}.pth".format(global_step)))
            # write data into tensorboard
            if global_step % 20 == 0:
                writer.add_scalar('loss/loss_all', loss_all, global_step)
                writer.add_scalar('loss/image_loss', image_loss, global_step)
                writer.add_scalar('loss/lpips_loss', lpips_loss, global_step)
                writer.add_scalar('loss/non_stamped_det_loss', non_stamped_det_loss, global_step)
                writer.add_scalar('loss/stamped_det_loss', stamped_det_loss, global_step)
                # writer.add_scalar('loss/loss_rtt', loss_rtt, global_step)
                # writer.add_scalar('loss/loss_crp', loss_crp, global_step)
                writer.add_scalar('loss/loss_rs', loss_rs, global_step)
            if global_step % 1000 == 0:
                writer.add_image('input/image_input', image_input[0], global_step)
                writer.add_image('encoded/stamped_image', stamped_image[0], global_step)
                writer.add_image('encoded/stamp_perturbation', stamp_perturbation[0]+0.5, global_step)
            
            global_step += 1
    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(detector, os.path.join(args.saved_models, "detector.pth"))
    # Evaluations
    evaluate(encoder, detector, test_loader, args)


if __name__ == '__main__':
    dataset_folder = './mirflickr/'
    if not os.path.exists(dataset_folder):
        os.system('hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xinghua/datasets/mirflickr25k.zip')
        os.system('unzip mirflickr25k.zip')
        os.system('rm -f mirflickr25k.zip')

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
    parser.add_argument('--std', type=float, default=0.1)
    user_args = parser.parse_args()
    main(vars(user_args))
