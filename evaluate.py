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

def evaluate(encoder, detector, test_loader, args):
    to_pil = torchvision.transforms.ToPILImage()
    eps_list = [4/255, 8/255, 16/255, 32/255, 64/255] # define the value of epsilon (contraint of noise scale)
    alpha_list = [1/255, 1/255, 1/255, 2/255, 2/255]
    
    adv_labels_stamp = torch.zeros(args.batch_size, 1, requires_grad=False).cuda()
    adv_labels_non   = torch.ones(args.batch_size, 1, requires_grad=False).cuda()
    
    for ind, eps in enumerate(eps_list):
        eval_path = './results/{}/evaluations/eps_{}'.format(args.run_name, int(eps*255))
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        save_name = 'eps_{}'.format(int(eps*255))
        # define the two attacks
        fgsm = FGSM_DEC(encoder, detector, eps)
        pgd = PGD_DEC(encoder, detector, eps=eps, alpha=alpha_list[ind], steps=40, random_start=True)
        
        ori_nonstamp_acc = 0
        ori_stamp_acc = 0
        ori_acc = 0
        rd_nonstamp_acc = 0
        rd_stamp_acc = 0
        rd_acc = 0
        fgsm_nonstamp_acc = 0
        fgsm_stamp_acc = 0
        fgsm_acc = 0
        pgd_nonstamp_acc = 0
        pgd_stamp_acc = 0
        pgd_acc = 0       
        rotate_nostp_acc = 0
        rotate_stp_acc = 0
        rotate_acc = 0
        center_crop_nostp_acc = 0
        center_crop_stp_acc = 0
        center_crop_acc = 0
        
        rotate_nostp_acc, rotate_stp_acc,rotate_acc,center_crop_nostp_acc,center_crop_stp_acc,center_crop_acc
        
        for batch_idx, (image_input, secret_input) in enumerate(test_loader, 0):
            image_input = image_input.cuda() # not necessary since it is already on GPU 
            secret_input = secret_input.cuda()
            IM = Traditional_Img_Manupulator(image_input)

            stamp_perturbation = encoder(image_input, secret_input)
            stamped_image = image_input + stamp_perturbation

            rand_pert = torch.randn_like(stamped_image).cuda()*0.5
            rand_pert = torch.clamp(rand_pert, -eps, eps)
            
            ## Original
            pred_ones_ori  = detector(image_input)
            ori_nonstamp_acc += torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_ori))).item()
            pred_zeros_ori = detector(stamped_image)
            ori_stamp_acc += torch.sum(torch.round(pred_zeros_ori).int()).item()
            ori_acc = ori_acc+ (torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_ori))).item()+torch.sum(torch.round(pred_zeros_ori).int()).item())
            
            ## Rotate
            degrees = [-20, 20]
            rtt_ori_image = IM.rotate(degrees)
            pred_ones_ori  = detector(rtt_ori_image)
            rotate_nostp_acc += torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_ori))).item()            
            rtt_stp_image = IM.rotate(stamped_image)
            pred_zeros_ori = detector(rtt_stp_image)
            rotate_stp_acc += torch.sum(torch.round(pred_zeros_ori).int()).item()
            rotate_acc = rotate_acc + (torch.sum(torch.round(pred_zeros_ori).int()).item()+ torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_ori))).item())
            
            ## Centercrop
            crop_range = [250, 400]
            cc_ori_image = IM.center_crop(crop_range)
            pred_ones_ori  = detector(cc_ori_image)
            center_crop_nostp_acc += torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_ori))).item()            
            cc_stp_image = IM.center_crop(stamped_image)
            pred_zeros_ori = detector(cc_stp_image)
            center_crop_stp_acc += torch.sum(torch.round(pred_zeros_ori).int()).item()
            center_crop_acc = center_crop_acc + (torch.sum(torch.round(pred_zeros_ori).int()).item()+ torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_ori))).item())
            
            
            ## Random perturbation
            pred_ones_rd  = detector(image_input+rand_pert)
            rd_nonstamp_acc += torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_rd))).item()            
            pred_zeros_rd = detector(stamped_image+rand_pert)
            rd_stamp_acc += torch.sum(torch.round(pred_zeros_rd).int()).item()
            rd_acc = rd_acc+ (torch.sum(torch.round(pred_zeros_rd).int()).item()+torch.sum(torch.round(torch.abs(adv_labels_non - pred_ones_rd))).item()      )
            
            ## FGSM perturbation
            fsgm_stamped_image_ori = fgsm(image_input, adv_labels_non)
            pred_ones_fgsm  = detector(fsgm_stamped_image_ori)
            fgsm_nonstamp_acc += torch.sum(torch.round(torch.abs(adv_labels_non-pred_ones_fgsm))).item()
            
            fsgm_stamped_image_ptb = fgsm(stamped_image, adv_labels_stamp)
            pred_zeros_fgsm  = detector(fsgm_stamped_image_ptb)
            fgsm_stamp_acc += torch.sum(torch.round(pred_zeros_fgsm).int()).item()  
            fgsm_acc =  fgsm_acc+(torch.sum(torch.round(pred_zeros_fgsm).int()).item() +torch.sum(torch.round(torch.abs(adv_labels_non-pred_ones_fgsm))).item())
            
            ## PGD perturbation
            pgd_stamped_image_ori = pgd(image_input, adv_labels_non)
            pred_ones_pgd  = detector(pgd_stamped_image_ori)
            pgd_nonstamp_acc += torch.sum(torch.round(torch.abs(adv_labels_non-pred_ones_pgd))).item()
            pgd_stamped_image_ptb = pgd(stamped_image, adv_labels_stamp)
            pred_zeros_pgd  = detector(pgd_stamped_image_ptb)
            pgd_stamp_acc += torch.sum(torch.round(pred_zeros_pgd).int()).item() 
            pgd_acc =  pgd_acc+(torch.sum(torch.round(torch.abs(adv_labels_non-pred_ones_pgd))).item()+torch.sum(torch.round(pred_zeros_pgd).int()).item() )
            
            # Save one image at each batch
            img = to_pil(image_input[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
#             plt.savefig('{}/orinal_{}.pdf'.format(eval_path, batch_idx),bbox_inches='tight')
            plt.savefig('{}/orinal_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=600)
            plt.close()
    
            img = to_pil(rtt_stp_image[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
#             plt.savefig('{}/rtt_{}.pdf'.format(eval_path, batch_idx),bbox_inches='tight')
            plt.savefig('{}/rtt_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=600)
            plt.close()
    
            img = to_pil(cc_stp_image[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
            plt.savefig('{}/cc_{}.pdf'.format(eval_path, batch_idx),bbox_inches='tight')
            plt.savefig('{}/cc_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=600)
            plt.close()

            img = to_pil((image_input+rand_pert)[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
#             plt.savefig('{}/rd_ptb_{}.pdf'.format(eval_path, batch_idx),bbox_inches='tight')
            plt.savefig('{}/rd_ptb_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=600)
            plt.close()

            img = to_pil((fsgm_stamped_image_ptb)[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
#             plt.savefig('{}/fgsm_ptb_{}.pdf'.format(eval_path, batch_idx),bbox_inches='tight')
            plt.savefig('{}/fgsm_ptb_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=600)
            plt.close()

            img = to_pil((pgd_stamped_image_ptb)[0].cpu())
            plt.figure()
            plt.axis('off')
            plt.imshow(img)
#             plt.savefig('{}/pgd_ptb_{}.pdf'.format(eval_path, batch_idx),bbox_inches='tight')
            plt.savefig('{}/pgd_ptb_{}.png'.format(eval_path, batch_idx),bbox_inches='tight', dpi=600)
            plt.close()
        ### log the evaluation results for each epsilon setting
        csv_log_file = './results/{}/evaluations/res.csv'.format(args.run_name)
        tn = len(test_loader)*args.batch_size
        values = [ori_nonstamp_acc/tn, ori_stamp_acc,ori_acc/(2*tn), rotate_nostp_acc/tn, rotate_stp_acc/tn,rotate_acc/(2*tn),center_crop_nostp_acc/tn,center_crop_stp_acc/tn,center_crop_acc/(2*tn), rd_nonstamp_acc/tn, rd_stamp_acc/tn,rd_acc/(2*tn), fgsm_nonstamp_acc/tn, fgsm_stamp_acc/tn, fgsm_acc/(2*tn), pgd_nonstamp_acc/tn, pgd_stamp_acc/tn, pgd_acc/(2*tn)]
        values = (1-np.array(values))*100
        values = values.tolist()
        if not os.path.exists(csv_log_file):
            with open(csv_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ['epsilon', 'step_size', 'ori_nonstamp_acc','ori_stamp_acc','ori_acc','rotate_nostp_acc', 'rotate_stp_acc','rotate_acc','center_crop_nostp_acc','center_crop_stp_acc','center_crop_acc','rd_nonstamp_acc','rd_stamp_acc', 'rd_acc', 'fgsm_nonstamp_acc', 
    'fgsm_stamp_acc', 'fgsm_acc','pgd_nonstamp_acc', 'pgd_stamp_acc', 'pgd_acc']
                writer.writerow(header)
                data = [0, 0]+[100]*18
                writer.writerow(data)
                f.close()
        else:
            with open(csv_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                log_line = [int(eps*255), int(alpha_list[ind]*255)] + values
                writer.writerow(log_line)
                f.close()
                
    run_name = args.run_name
    file_path = "./results/{}/evaluations/res.csv".format(run_name)
    df = pd.read_csv(file_path)

    fig_path = "./results/{}/evaluations/figures".format(run_name)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig_names = ['original','rotate', 'center_crop', 'randn_ptb', 'fgsm_ptb','pgd_ptb']
    header = ['ori_nonstamp_acc','ori_stamp_acc','ori_acc','rotate_nostp_acc', 'rotate_stp_acc','rotate_acc','center_crop_nostp_acc','center_crop_stp_acc','center_crop_acc','rd_nonstamp_acc','rd_stamp_acc', 'rd_acc', 'fgsm_nonstamp_acc', 
    'fgsm_stamp_acc', 'fgsm_acc','pgd_nonstamp_acc', 'pgd_stamp_acc', 'pgd_acc']
    for i in range(len(fig_names)):
            fig_name = fig_names[i]
            plt.style.use('seaborn-bright')
            plt.figure()
            df.plot(x="epsilon", y=[header[i*3], header[i*3+1], header[i*3+2]], kind="bar", legend=False)
            plt.ylabel('Detection Accuracy')
            plt.xlabel('Epsilon')
            plt.title(fig_names[i], x=0.5, y=1.1)
            plt.xticks([0,1,2,3,4],['{}/255'.format(x) for x in df['epsilon'].values.tolist()], rotation = 0) 
            plt.legend(['Non_Stamped', 'Stamped','Average'], ncol=3, bbox_to_anchor=(0.45, 0.62,0.5, 0.5))
            plt.savefig('{}/{}.png'.format(fig_path, fig_name),bbox_inches='tight', dpi=500)