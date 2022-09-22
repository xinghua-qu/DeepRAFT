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

def acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, std):
    repeat_times = 1000
    total_non_stmp_acc = 0
    total_stmp_acc     = 0
    for i in range(repeat_times):
        with torch.no_grad():
            rs_img_non_stmp = img_non_stmp +  torch.randn_like(img_non_stmp)*std
            rs_img_stmp     = img_stmp +  torch.randn_like(img_stmp)*std
            preds  = detector(rs_img_non_stmp)
            pred_ones_ori = torch.argmax(preds, dim=1)

            preds = detector(rs_img_stmp)
            pred_zeros_ori = torch.argmax(preds, dim=1)

            non_stmp_acc = (pred_ones_ori.unsqueeze(1)  ==ones).sum().item()/img_non_stmp.size()[0]
            stmp_acc     = (pred_zeros_ori.unsqueeze(1) ==zeros).sum().item()/img_non_stmp.size()[0]
            total_non_stmp_acc += non_stmp_acc
            total_stmp_acc     += stmp_acc
    total_non_stmp_acc = total_non_stmp_acc/repeat_times
    total_stmp_acc     = total_stmp_acc/repeat_times
    return total_non_stmp_acc, total_stmp_acc


def evaluate(encoder, detector, test_loader, args):
    to_pil = torchvision.transforms.ToPILImage()
    eps_list = [2/255, 4/255, 8/255, 16/255, 32/255, 64/255] # define the value of epsilon (contraint of noise scale)
    alpha_list = [1/255, 1/255, 2/255, 4/255, 6/255, 8/255]
    
    zeros  = torch.zeros(args.batch_size,  1, requires_grad=False, dtype = int).cuda()
    ones   = torch.ones (args.batch_size,  1, requires_grad=False, dtype = int).cuda()
    
    scale_1 = np.random.uniform(0, 0.4)
    degree_scale = np.random.randint(0,45)
    cropper = T.RandomResizedCrop(size=(400, 400), scale=(scale_1, 1))
    rotater = T.RandomRotation(degrees=(-degree_scale, degree_scale))
    jitter = T.ColorJitter(brightness=scale_1, contrast=scale_1, saturation=scale_1, hue=scale_1)
    blurrer = T.GaussianBlur(kernel_size=(3, 7), sigma=(1, 3))
    perspective_transformer = T.RandomPerspective(distortion_scale=scale_1, p=1.0)
    jpeg = DiffJPEG(height=400, width=400, differentiable=True, quality=int(np.random.uniform(50,100,1)))
    jpeg = jpeg.cuda()
    
    eval_path = './results/{}/evaluations/traditional/'.format(args.run_name)
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
        
    ori_nonstamp_acc = 0
    ori_stamp_acc = 0
    ori_acc = 0 
    rtt_nonstamp_acc = 0
    rtt_stamp_acc = 0
    rtt_acc = 0
    crp_nonstamp_acc = 0
    crp_stamp_acc = 0
    crp_acc = 0
    wrp_nonstamp_acc = 0
    wrp_stamp_acc = 0
    wrp_acc = 0
    blr_nonstamp_acc = 0
    blr_stamp_acc = 0
    blr_acc = 0
    jtr_nonstamp_acc = 0
    jtr_stamp_acc = 0
    jtr_acc = 0
    jpg_nonstamp_acc = 0
    jpg_stamp_acc = 0
    jpg_acc = 0
    mix_nonstamp_acc = 0
    mix_stamp_acc = 0
    mix_acc = 0
    
    for batch_idx, (image_input) in enumerate(test_loader, 0):
        image_input = image_input.cuda() # not necessary since it is already on GPU 
        stamp_perturbation = encoder(image_input )
        stamped_image = image_input + stamp_perturbation
           
        ## Original accuracy
        img_non_stmp = image_input
        img_stmp     = stamped_image
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        ori_nonstamp_acc += non_stmp_acc
        ori_stamp_acc    += stmp_acc
        ori_acc          += (non_stmp_acc + stmp_acc)/2
        print('ori',non_stmp_acc,stmp_acc)
        
        img = to_pil(image_input[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/original_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close()
        img = to_pil(stamped_image[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/original_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close()
        
        ## Coulor jitter
        img_non_stmp = jitter(image_input)
        img_stmp     = jitter(stamped_image)
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        jtr_nonstamp_acc += non_stmp_acc
        jtr_stamp_acc    += stmp_acc
        jtr_acc          += (non_stmp_acc + stmp_acc)/2
        print('jtr',non_stmp_acc,stmp_acc)
        
        img = to_pil(img_stmp[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/jtr_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close()
        
        ## Rotate
        img_non_stmp = rotater(image_input)
        img_stmp     = rotater(stamped_image)
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        rtt_nonstamp_acc += non_stmp_acc
        rtt_stamp_acc    += stmp_acc
        rtt_acc          += (non_stmp_acc + stmp_acc)/2
        print('rtt',non_stmp_acc,stmp_acc)
        
        img = to_pil(img_stmp[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/rtt_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close()
        
        ## Gaussian  blur
        img_non_stmp  = blurrer(image_input)
        img_stmp      = blurrer(stamped_image)
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        blr_nonstamp_acc += non_stmp_acc
        blr_stamp_acc    += stmp_acc
        blr_acc          += (non_stmp_acc + stmp_acc)/2
        print('blr',non_stmp_acc,stmp_acc)
        
        img = to_pil(img_stmp[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/blr_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close()
        
        ## Crop (random)        
        img_non_stmp  = cropper(image_input)
        img_stmp      = cropper(stamped_image)
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        crp_nonstamp_acc += non_stmp_acc
        crp_stamp_acc    += stmp_acc
        crp_acc          += (non_stmp_acc + stmp_acc)/2
        print('crp',non_stmp_acc,stmp_acc)
        
        img = to_pil(img_stmp[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/crp_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close()
        
        ## Warping (simulate the shape change in rephotography)
        img_non_stmp  = perspective_transformer(image_input)
        img_stmp      = perspective_transformer(stamped_image)
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        wrp_nonstamp_acc += non_stmp_acc
        wrp_stamp_acc    += stmp_acc
        wrp_acc          += (non_stmp_acc + stmp_acc)/2
        print('wrp',non_stmp_acc,stmp_acc)
        
        img = to_pil(img_stmp[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/wrp_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close() 
        
        
        ## Warping (simulate the shape change in rephotography)
        img_non_stmp  = jpeg(image_input)
        img_stmp      = jpeg(stamped_image)
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        jpg_nonstamp_acc += non_stmp_acc
        jpg_stamp_acc    += stmp_acc
        jpg_acc          += (non_stmp_acc + stmp_acc)/2
        print('jpg',non_stmp_acc,stmp_acc)
        
        img = to_pil(img_stmp[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/wrp_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close() 
        
        ## Mixed (simulate the shape change in rephotography)
        new_image = image_input
        if np.random.rand()<=args.crp_dropout: new_image = jitter(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = cropper(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = blurrer(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = rotater(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = perspective_transformer(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = jpeg(new_image)
        img_non_stmp = new_image
        
        new_image = stamped_image
        if np.random.rand()<=args.crp_dropout: new_image = jitter(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = cropper(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = blurrer(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = rotater(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = perspective_transformer(new_image)
        if np.random.rand()<=args.crp_dropout: new_image = jpeg(new_image)
        img_stmp = new_image
        
        non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
        mix_nonstamp_acc += non_stmp_acc
        mix_stamp_acc    += stmp_acc
        mix_acc          += (non_stmp_acc + stmp_acc)/2
        print('mix',non_stmp_acc,stmp_acc)
        
        img = to_pil(img_stmp[0].cpu())
        plt.figure()
        plt.axis('off')
        plt.imshow(img)
        plt.savefig('{}/wrp_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
        plt.close()
    
    ### log the evaluation results for each epsilon setting
    csv_log_file = '{}res.csv'.format(eval_path)
    tn = batch_idx+1
    values = [ori_nonstamp_acc,ori_stamp_acc,ori_acc,rtt_nonstamp_acc, rtt_stamp_acc, rtt_acc, crp_nonstamp_acc, crp_stamp_acc,crp_acc,wrp_nonstamp_acc,wrp_stamp_acc,wrp_acc, blr_nonstamp_acc, blr_stamp_acc,blr_acc,jtr_nonstamp_acc,jtr_stamp_acc,jtr_acc, jpg_nonstamp_acc,jpg_stamp_acc,jpg_acc, mix_nonstamp_acc, mix_stamp_acc, mix_acc]
#     values = [ori_acc, rtt_acc, crp_acc, wrp_acc, blr_acc, jtr_acc]
    values = (np.array(values)/tn)*100
    values = values.tolist()
    if not os.path.exists(csv_log_file):
        with open(csv_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['ori_nonstamp_acc','ori_stamp_acc','ori_acc','rtt_nonstamp_acc', 'rtt_stamp_acc', 'rtt_acc', 'crp_nonstamp_acc', 'crp_stamp_acc','crp_acc','wrp_nonstamp_acc','wrp_stamp_acc','wrp_acc', 'blr_nonstamp_acc', 'blr_stamp_acc','blr_acc','jtr_nonstamp_acc','jtr_stamp_acc','jtr_acc', 'jpg_nonstamp_acc','jpg_stamp_acc','jpg_acc', 'mix_nonstamp_acc','mix_stamp_acc','mix_acc']
            writer.writerow(header)
            writer.writerow(values)
            f.close()
    else:
        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            header = ['ori_nonstamp_acc','ori_stamp_acc','ori_acc','rtt_nonstamp_acc', 'rtt_stamp_acc', 'rtt_acc', 'crp_nonstamp_acc', 'crp_stamp_acc','crp_acc','wrp_nonstamp_acc','wrp_stamp_acc','wrp_acc', 'blr_nonstamp_acc', 'blr_stamp_acc','blr_acc','jtr_nonstamp_acc','jtr_stamp_acc','jtr_acc']
            writer.writerow(header)
            writer.writerow(values)
            f.close()
    ## plot the results from traditional evaluations
    plt_values = [ori_acc, rtt_acc, crp_acc, wrp_acc, blr_acc, jtr_acc, jpg_acc, mix_acc]
    plt_values = (np.array(plt_values)/tn)*100
    my_xticks = ['Original', 'Rotation', 'Cropping', 'Warping', 'Gaussian Blur', 'Color Filter', 'JPEG Compression', 'Mixed Corruptions']
    plt.style.use('seaborn-bright')
    plt.figure()
    plt.bar(range(len(plt_values)), plt_values)
    plt.ylabel('Evaluation Method')
    plt.xlabel('Accuracy')
    plt.xticks([i for i in range(len(my_xticks))], my_xticks, rotation = 45)
    plt.savefig('./results/{}/evaluations/Traditional.png'.format(args.run_name),bbox_inches='tight', dpi=400)
    
    ######################################### Adversarial Evaluations ################################
    rd_nonstamp_acc = [0]*len(eps_list)
    rd_stamp_acc = [0]*len(eps_list)
    rd_acc = [0]*len(eps_list)
    fgsm_nonstamp_acc = [0]*len(eps_list)
    fgsm_stamp_acc = [0]*len(eps_list)
    fgsm_acc = [0]*len(eps_list)
    pgd_nonstamp_acc = [0]*len(eps_list)
    pgd_stamp_acc = [0]*len(eps_list)
    pgd_acc = [0]*len(eps_list)  
    sqr_nonstamp_acc = [0]*len(eps_list)
    sqr_stamp_acc = [0]*len(eps_list)
    sqr_acc = [0]*len(eps_list) 
    pgdce_nonstamp_acc = [0]*len(eps_list)
    pgdce_stamp_acc = [0]*len(eps_list)
    pgdce_acc = [0]*len(eps_list) 
    auto_nonstamp_acc = [0]*len(eps_list)
    auto_stamp_acc = [0]*len(eps_list)
    auto_acc = [0]*len(eps_list) 
    automix_nonstamp_acc = [0]*len(eps_list)
    automix_stamp_acc = [0]*len(eps_list)
    automix_acc = [0]*len(eps_list) 
    
    ones_labels = torch.FloatTensor([0,1])
    ones_labels = ones_labels.repeat(args.batch_size,1).cuda()
    zeros_labels = torch.FloatTensor([1,0])
    zeros_labels = zeros_labels.repeat(args.batch_size,1).cuda()
    
    for ind, eps in enumerate(eps_list):
        eval_path = './results/{}/evaluations/adversarial/eps_{}'.format(args.run_name, int(eps*255))
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
            
        # define the FGSM and PGD attacks
        ones  = ones.cuda()
        zeros = zeros.cuda()
        fgsm        = FGSM_DEC(detector, eps)
        pgd         = PGD_DEC(detector, eps, alpha=alpha_list[ind], steps=40, random_start=True)        
        for batch_idx, (image_input ) in enumerate(test_loader, 0):
            with torch.no_grad():
                image_input = image_input.cuda()
                stamp_perturbation = encoder(image_input )            
                stamped_image = image_input + stamp_perturbation

                ## random perturbation
                img_non_stmp = torch.clamp(image_input + torch.randn_like(image_input, device='cuda') * eps, -1, 1)
                img_stmp     = torch.clamp(stamped_image + torch.randn_like(stamped_image, device='cuda') * eps, -1, 1)

                zeros_lb  = torch.zeros(args.batch_size, 1,  requires_grad=False, dtype = int)
                ones_lb   = torch.ones (args.batch_size, 1,  requires_grad=False, dtype = int)
            non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp.cuda(), img_stmp.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
            rd_nonstamp_acc[ind] += non_stmp_acc
            rd_stamp_acc[ind]    += stmp_acc
            rd_acc[ind]          += (non_stmp_acc + stmp_acc)/2

            img = to_pil(img_stmp[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('{}/rd_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
            plt.close()

            ## FGSM perturbation
            img_non_stmp = fgsm(image_input, ones_labels)
            img_stmp     = fgsm(stamped_image, zeros_labels)

            zeros_lb  = torch.zeros(args.batch_size, 1,  requires_grad=False, dtype = int)
            ones_lb   = torch.ones (args.batch_size, 1,  requires_grad=False, dtype = int)
            non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp.cuda(), img_stmp.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
            fgsm_nonstamp_acc[ind] += non_stmp_acc
            fgsm_stamp_acc[ind]    += stmp_acc
            fgsm_acc[ind]          += (non_stmp_acc + stmp_acc)/2

            img = to_pil(img_stmp[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('{}/fgsm_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
            plt.close()

            ## PGD perturbation
            img_non_stmp = pgd(image_input, ones_labels)
            img_stmp     = pgd(stamped_image, zeros_labels)

            zeros_lb  = torch.zeros(args.batch_size, 1,  requires_grad=False, dtype = int)
            ones_lb   = torch.ones (args.batch_size, 1,  requires_grad=False, dtype = int)
            non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp.cuda(), img_stmp.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
            pgd_nonstamp_acc[ind] += non_stmp_acc
            pgd_stamp_acc[ind]    += stmp_acc
            pgd_acc[ind]          += (non_stmp_acc + stmp_acc)/2

            img = to_pil(img_stmp[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('{}/pgd_stamped_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=300)
            plt.close() 
        rd_nonstamp_acc[ind] = rd_nonstamp_acc[ind]/(batch_idx+1)
        rd_stamp_acc[ind] = rd_stamp_acc[ind]/(batch_idx+1)
        rd_acc[ind] = rd_acc[ind]/(batch_idx+1)
        
        fgsm_nonstamp_acc[ind] = fgsm_nonstamp_acc[ind]/(batch_idx+1)
        fgsm_stamp_acc[ind] = fgsm_stamp_acc[ind]/(batch_idx+1)
        fgsm_acc[ind] = fgsm_acc[ind]/(batch_idx+1)
        
        pgd_nonstamp_acc[ind] = pgd_nonstamp_acc[ind]/(batch_idx+1)
        pgd_stamp_acc[ind] = pgd_stamp_acc[ind]/(batch_idx+1)
        pgd_acc[ind] = pgd_acc[ind]/(batch_idx+1)
        
        ## pgd-ce attack
        with torch.no_grad():
            ### auto attack [square + apg-ce]
            auto_attack = AutoAttack(detector, norm='Linf', eps=eps, version='standard')
            auto_attack.attacks_to_run = [ 'apgd-ce']
            image_inputs = [image_input for (image_input ) in test_loader]
            x_test = torch.cat(image_inputs, 0)

            x_stmp_test = []
            for i in range(x_test.size()[0]//args.batch_size):
                image_input  = x_test[i*args.batch_size:(i+1)*args.batch_size].cuda()
                 
                stamp_perturbation = encoder(image_input.cuda() )            
                stamped_image = image_input + stamp_perturbation
                x_stmp_test.append(stamped_image)
            x_stmp_test = torch.cat(x_stmp_test, 0)

            zeros  = torch.zeros(x_stmp_test.size()[0],   requires_grad=False, dtype = int)
            ones   = torch.ones (x_stmp_test.size()[0],   requires_grad=False, dtype = int)
            adv_stmp     = auto_attack.run_standard_evaluation(x_stmp_test.cuda(), zeros.cuda(), bs=args.batch_size)
            adv_non_stmp = auto_attack.run_standard_evaluation(x_test.cuda(), ones.cuda(), bs=args.batch_size)
            for i1 in range(adv_non_stmp.size()[0]//args.batch_size):
                img_non_stmp_adv  = adv_non_stmp[i*args.batch_size:(i+1)*args.batch_size]
                img_stmp_adv = adv_stmp[i*args.batch_size:(i+1)*args.batch_size]
                zeros_lb  = torch.zeros(args.batch_size, 1,  requires_grad=False, dtype = int)
                ones_lb   = torch.ones (args.batch_size, 1,  requires_grad=False, dtype = int)
                non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp_adv.cuda(), img_stmp_adv.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
                pgdce_nonstamp_acc[ind] += non_stmp_acc
                pgdce_stamp_acc[ind]    += stmp_acc
                pgdce_acc[ind]          += (non_stmp_acc + stmp_acc)/2
                img = to_pil(img_stmp_adv[0].cpu())
                plt.figure()
                plt.axis('off')
                plt.imshow(img)
                plt.savefig('{}/apgd-ce_stamped_{}.png'.format(eval_path, i1),bbox_inches='tight', dpi=300)
                plt.close()
            print(pgdce_nonstamp_acc[ind], pgdce_stamp_acc[ind], pgdce_acc[ind])
            pgdce_nonstamp_acc[ind] = pgdce_nonstamp_acc[ind]/(i1+1)
            pgdce_stamp_acc[ind]    = pgdce_stamp_acc[ind]/(i1+1)
            pgdce_acc[ind]          = pgdce_acc[ind]/(i1+1)
            print(pgdce_nonstamp_acc[ind], pgdce_stamp_acc[ind], pgdce_acc[ind])
            print(non_stmp_acc, stmp_acc, (non_stmp_acc + stmp_acc)/2, i1, pgdce_nonstamp_acc[ind], pgdce_stamp_acc[ind], pgdce_acc[ind] )
        
        ## square attack
        with torch.no_grad():
            ### auto attack [square + apg-ce]
            auto_attack = AutoAttack(detector, norm='Linf', eps=eps, version='standard')
            auto_attack.attacks_to_run = [ 'square']
            image_inputs = [image_input for (image_input ) in test_loader]
            x_test = torch.cat(image_inputs, 0)

            x_stmp_test = []
            for i in range(x_test.size()[0]//args.batch_size):
                image_input  = x_test[i*args.batch_size:(i+1)*args.batch_size].cuda()
                 
                stamp_perturbation = encoder(image_input.cuda())            
                stamped_image = image_input + stamp_perturbation
                x_stmp_test.append(stamped_image)
            x_stmp_test = torch.cat(x_stmp_test, 0)

            zeros  = torch.zeros(x_stmp_test.size()[0],   requires_grad=False, dtype = int)
            ones   = torch.ones (x_stmp_test.size()[0],   requires_grad=False, dtype = int)
            adv_stmp     = auto_attack.run_standard_evaluation(x_stmp_test, zeros.cuda(), bs=args.batch_size)
            adv_non_stmp = auto_attack.run_standard_evaluation(x_test.cuda(), ones.cuda(), bs=args.batch_size)
            for i1 in range(adv_non_stmp.size()[0]//args.batch_size):
                img_non_stmp_adv  = adv_non_stmp[i*args.batch_size:(i+1)*args.batch_size]
                img_stmp_adv      = adv_stmp[i*args.batch_size:(i+1)*args.batch_size]
                zeros_lb  = torch.zeros(args.batch_size, 1,  requires_grad=False, dtype = int)
                ones_lb   = torch.ones (args.batch_size, 1,  requires_grad=False, dtype = int)
                non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp_adv.cuda(), img_stmp_adv.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
                sqr_nonstamp_acc[ind] += non_stmp_acc
                sqr_stamp_acc[ind]    += stmp_acc
                sqr_acc[ind]          += (non_stmp_acc + stmp_acc)/2
                img = to_pil(img_stmp_adv[0].cpu())
                plt.figure()
                plt.axis('off')
                plt.imshow(img)
                plt.savefig('{}/sqr_stamped_{}.png'.format(eval_path, i1),bbox_inches='tight', dpi=300)
                plt.close()
            sqr_nonstamp_acc[ind] = sqr_nonstamp_acc[ind]/(i1+1)
            sqr_stamp_acc[ind]    = sqr_stamp_acc[ind]/(i1+1)
            sqr_acc[ind]          = sqr_acc[ind]/(i1+1)
        
        ## auto attack
        with torch.no_grad():
            ### auto attack [square + apg-ce]
            auto_attack = AutoAttack(detector, norm='Linf', eps=eps, version='standard')
            auto_attack.attacks_to_run = [ 'square', 'apgd-ce']
            image_inputs = [image_input for (image_input ) in test_loader]
            x_test = torch.cat(image_inputs, 0)

            x_stmp_test = []
            for i in range(x_test.size()[0]//args.batch_size):
                image_input  = x_test[i*args.batch_size:(i+1)*args.batch_size].cuda()                 
                stamp_perturbation = encoder(image_input.cuda() )            
                stamped_image = image_input + stamp_perturbation
                x_stmp_test.append(stamped_image)
            x_stmp_test = torch.cat(x_stmp_test, 0)

            zeros  = torch.zeros(x_stmp_test.size()[0],   requires_grad=False, dtype = int)
            ones   = torch.ones (x_stmp_test.size()[0],   requires_grad=False, dtype = int)
            adv_stmp     = auto_attack.run_standard_evaluation(x_stmp_test, zeros.cuda(), bs=args.batch_size)
            adv_non_stmp = auto_attack.run_standard_evaluation(x_test.cuda(), ones.cuda(), bs=args.batch_size)
            for i1 in range(adv_non_stmp.size()[0]//args.batch_size):
                img_non_stmp_adv  = adv_non_stmp[i*args.batch_size:(i+1)*args.batch_size]
                img_stmp_adv = adv_stmp[i*args.batch_size:(i+1)*args.batch_size]
                zeros_lb  = torch.zeros(args.batch_size, 1,  requires_grad=False, dtype = int)
                ones_lb   = torch.ones (args.batch_size, 1,  requires_grad=False, dtype = int)
                non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp_adv.cuda(), img_stmp_adv.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
                auto_nonstamp_acc[ind] += non_stmp_acc
                auto_stamp_acc[ind]    += stmp_acc
                auto_acc[ind]          += (non_stmp_acc + stmp_acc)/2
                img = to_pil(img_stmp_adv[0].cpu())
                plt.figure()
                plt.axis('off')
                plt.imshow(img)
                plt.savefig('{}/auto_stamped_{}.png'.format(eval_path, i1),bbox_inches='tight', dpi=300)
                plt.close()
            auto_nonstamp_acc[ind] = auto_nonstamp_acc[ind]/(i1+1)
            auto_stamp_acc[ind]    = auto_stamp_acc[ind]/(i1+1)
            auto_acc[ind]          = auto_acc[ind]/(i1+1)
            print(non_stmp_acc, stmp_acc, (non_stmp_acc + stmp_acc)/2, i1, auto_nonstamp_acc[ind], auto_stamp_acc[ind], auto_acc[ind] )
        
        ## auto mixed attack
        with torch.no_grad():
            ## use previous obtained auto attacks
            for i1 in range(adv_non_stmp.size()[0]//args.batch_size):
                img_non_stmp_adv  = adv_non_stmp[i*args.batch_size:(i+1)*args.batch_size]
                img_stmp_adv      = adv_stmp[i*args.batch_size:(i+1)*args.batch_size]
                
                new_image = img_non_stmp_adv
                if np.random.rand()<=args.crp_dropout: new_image = jitter(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = cropper(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = blurrer(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = rotater(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = perspective_transformer(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = jpeg(new_image.cuda())
                img_non_stmp_adv = new_image

                new_image = img_stmp_adv
                if np.random.rand()<=args.crp_dropout: new_image = jitter(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = cropper(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = blurrer(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = rotater(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = perspective_transformer(new_image)
                if np.random.rand()<=args.crp_dropout: new_image = jpeg(new_image)
                img_stmp_adv = new_image   
                
                zeros_lb  = torch.zeros(args.batch_size, 1,  requires_grad=False, dtype = int)
                ones_lb   = torch.ones (args.batch_size, 1,  requires_grad=False, dtype = int)
                non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp_adv.cuda(), img_stmp_adv.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
                automix_nonstamp_acc[ind] += non_stmp_acc
                automix_stamp_acc[ind]    += stmp_acc
                automix_acc[ind]          += (non_stmp_acc + stmp_acc)/2
                img = to_pil(img_stmp_adv[0].cpu())
                plt.figure()
                plt.axis('off')
                plt.imshow(img)
                plt.savefig('{}/automix_stamped_{}.png'.format(eval_path, i1),bbox_inches='tight', dpi=300)
                plt.close()
            automix_nonstamp_acc[ind] = automix_nonstamp_acc[ind]/(i1+1)
            automix_stamp_acc[ind]    = automix_stamp_acc[ind]/(i1+1)
            automix_acc[ind]          = automix_acc[ind]/(i1+1)

        ### log the evaluation results for each epsilon setting
        csv_log_file = './results/{}/evaluations/adversarial/res.csv'.format(args.run_name)
        values = [rd_nonstamp_acc[ind],rd_stamp_acc[ind],rd_acc[ind], fgsm_nonstamp_acc[ind],fgsm_stamp_acc[ind],fgsm_acc[ind], pgd_nonstamp_acc[ind],pgd_stamp_acc[ind],pgd_acc[ind], sqr_nonstamp_acc[ind],sqr_stamp_acc[ind],sqr_acc[ind], pgdce_nonstamp_acc[ind],pgdce_stamp_acc[ind],pgdce_acc[ind], auto_nonstamp_acc[ind],auto_stamp_acc[ind],auto_acc[ind], automix_nonstamp_acc[ind],automix_stamp_acc[ind],automix_acc[ind]]
        values = (np.array(values))*100
        values = values.tolist()
        if not os.path.exists(csv_log_file):
            with open(csv_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['epsilon', 'step_size', 'rd_nonstamp_acc','rd_stamp_acc','rd_acc', 'fgsm_nonstamp_acc','fgsm_stamp_acc','fgsm_acc', 'pgd_nonstamp_acc','pgd_stamp_acc','pgd_acc', 'sqr_nonstamp_acc','sqr_stamp_acc','sqr_acc', 'pgdce_nonstamp_acc','pgdce_stamp_acc','pgdce_acc', 'auto_nonstamp_acc','auto_stamp_acc','auto_acc', 'automix_nonstamp_acc','automix_stamp_acc','automix_acc']
                writer.writerow(header)
                data = [0, 0]+[100]*(len(header)-2)
                writer.writerow(data)
                f.close()
        with open(csv_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            log_line = [int(eps*255), int(alpha_list[ind]*255)] + values
            writer.writerow(log_line)
            f.close()
                
    df = pd.read_csv(csv_log_file)

    fig_path = "./results/{}/evaluations/adversarial/figures".format(args.run_name)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_names = ['randn_ptb', 'fgsm_ptb','pgd_ptb', 'square_attack', 'auto_attack', 'auto_mix']
    header = ['rd_nonstamp_acc','rd_stamp_acc','rd_acc', 'fgsm_nonstamp_acc','fgsm_stamp_acc','fgsm_acc', 'pgd_nonstamp_acc','pgd_stamp_acc','pgd_acc', 'sqr_nonstamp_acc','sqr_stamp_acc','sqr_acc','pgdce_nonstamp_acc','pgdce_stamp_acc','pgdce_acc', 'auto_nonstamp_acc','auto_stamp_acc','auto_acc', 'automix_nonstamp_acc','automix_stamp_acc','automix_acc']    
    for i in range(len(fig_names)):
        fig_name = fig_names[i]
        plt.style.use('seaborn-bright')
        plt.figure()
        df.plot(x="epsilon", y=[header[i*3], header[i*3+1], header[i*3+2]], kind="bar", legend=False)
        plt.ylabel('Detection Accuracy')
        plt.xlabel('Epsilon')
        plt.title(fig_names[i], x=0.5, y=1.1)
        stick_data = df['epsilon'].values.tolist()
        plt.xticks(np.arange(len(stick_data)),['{}/255'.format(x) for x in stick_data], rotation = 0) 
        plt.legend(['Non_Stamped', 'Stamped','Average'], ncol=3, bbox_to_anchor=(0.45, 0.62,0.5, 0.5))
        plt.savefig('{}/{}.png'.format(fig_path, fig_name),bbox_inches='tight', dpi=500)