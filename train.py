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

def main(user_args):
    
    # get the argument
    with open('setting.yaml', 'r') as f:
        fixed_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    args = EasyDict(dict( user_args, **fixed_args ))
    
    args.saved_models = './results/{}/{}'.format(args.run_name, args.saved_models)
    if not os.path.exists(args.saved_models):
        os.makedirs(args.saved_models)
        
    args.logs_path = './results/{}/{}'.format(args.run_name, args.logs_path)
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
        
    # Prepare the tensorboard: whether on local machine or Arnold cloud (hdfs savings)
    if args.tensorboard==0:
        log_path = os.path.join(args.logs_path, 'tensorboard')
        writer = SummaryWriter(log_path)
    else:
        HDFS_log_dir = os.environ.get("ARNOLD_OUTPUT")
        if HDFS_log_dir:  # if Arnold supports remote Tensorboard
            log_path = f'{HDFS_log_dir}/logs/{args.run_name}'
            cmd = f'hdfs dfs -mkdir -p {log_path}'
            print(cmd)
            os.system(cmd)
        writer = SummaryWriter(log_path)  # this line alone will create the folder
    
    dataset = StegaData(args.train_path, args.secret_size, size=(400, 400))
    

    datasets = train_test_dataset(dataset, test_split=0.2)
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)

    # Create the models
    encoder = model.StegaStampEncoder()
    detector = model.StampDetector()
    lpips_alex = lpips.LPIPS(net="alex", verbose=False)
    if args.cuda:
        encoder = encoder.cuda()
        detector = detector.cuda()
        lpips_alex.cuda()
    
    encoder.train()
    detector.train()
    lpips_alex.train()
    
    opt_vars = [{'params': encoder.parameters()},
              {'params': detector.parameters()}]

    optimizer = optim.Adam(opt_vars, lr=args.lr)
    
    global_step = 0
    ones = torch.ones(args.batch_size, 1, requires_grad=False).cuda() #label for non stamped image
    zeros = torch.ones(args.batch_size, 1, requires_grad=False).cuda()*0 # label for stamped image
    for epoch in range(args.max_epoch):      
        for batch_idx, (image_input, secret_input) in enumerate(train_loader, 0):
            image_input = image_input.cuda() # not necessary since it is already on GPU 
            secret_input = secret_input.cuda()
            stamp_perturbation = encoder(image_input, secret_input)
            stamped_image = image_input + stamp_perturbation
            
            predicted_ones  = detector(image_input)
            predicted_zeros = detector(stamped_image)
            
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
            non_stamped_det_loss = F.binary_cross_entropy(predicted_ones, ones)
            stamped_det_loss = F.binary_cross_entropy(predicted_zeros, zeros)
            
            loss_all = image_loss + lpips_loss + non_stamped_det_loss + stamped_det_loss
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            if global_step % args.log_interval == 0:
                name_space = ['image_loss:' , 'lpips_loss:' , 'non_stamp_loss:' , 'stamp_loss:']
                losses_log = [image_loss.item() , lpips_loss.item() , non_stamped_det_loss.item() , stamped_det_loss.item()]
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
            if global_step % 500 == 0:
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
    parser.add_argument('--max_epoch', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--l2_loss_scale', type=float, default=2)
    parser.add_argument('--lr', type=float, default=0.00005)
    user_args = parser.parse_args()
    main(vars(user_args))
