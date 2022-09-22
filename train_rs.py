import os
import yaml
import random
import model
import numpy as np
from glob import glob
from easydict import EasyDict
from PIL import Image
from torch import optim
import utils
from dataset import StegaData, train_test_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import lpips
import argparse
from kornia import color
from evaluate_rs import evaluate, acc_calculate
import matplotlib.pyplot as plt 
from DiffJPEG.DiffJPEG import DiffJPEG
import torchvision.transforms as T
from autoattack import AutoAttack

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
    log_path = os.path.join(args.logs_path, 'tensorboard')
    writer = SummaryWriter(log_path)
   
    ## dataset
    dataset = StegaData(args.train_path, size=(400, 400))   
    datasets = train_test_dataset(dataset, test_split=0.1)
    train_val_dataset = datasets['train']    
    subsets = train_test_dataset(train_val_dataset, test_split=0.02)
    
    train_dataset = subsets['train']
    val_dataset   = subsets['test']
    test_dataset  = datasets['test']
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True,  pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    
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
    lpips_alex.eval()
    
    ## losses
    celoss  = torch.nn.CrossEntropyLoss()
    l1_loss = torch.nn.L1Loss()
    
    ## initialization
    global_step = 0
    ones = torch.FloatTensor([0,1])
    ones = ones.repeat(args.batch_size,1).cuda()
    zeros = torch.FloatTensor([1,0])
    zeros = zeros.repeat(args.batch_size,1).cuda()
    
    opt_vars = [{'params': encoder.parameters()},
              {'params': detector.parameters()}]

    optimizer = optim.Adam(opt_vars, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999875)
   
    global_step = 0
    for epoch in range(args.max_epoch):    
        for batch_idx, (image_input) in enumerate(train_loader, 0):
            image_input = image_input.cuda() # not necessary since it is already on GPU 
            stamp_perturbation = encoder(image_input)           
            stamped_image = image_input + stamp_perturbation 
            
            rs_image_input =  image_input + torch.randn_like(image_input)*args.std
            rs_stamped_image =  stamped_image + torch.randn_like(stamped_image)*args.std
            targets  = torch.cat((ones, zeros), dim=0)
            imputs   = torch.cat((rs_image_input, rs_stamped_image), dim=0)
            preds   = detector(imputs)
            loss_acc = celoss(preds, targets)
            
            # imperceptibility of stamp perturbation
            normalized_input = image_input * 2 - 1
            normalized_encoded = stamped_image * 2 - 1
            lpips_loss = torch.mean(lpips_alex(normalized_input, normalized_encoded)) 
            yuv_scales = [args.y_scale, args.u_scale, args.v_scale]
            encoded_image_yuv = color.rgb_to_yuv(stamped_image)
            image_input_yuv = color.rgb_to_yuv(image_input)
            im_diff = encoded_image_yuv - image_input_yuv
            yuv_loss = torch.mean((im_diff) ** 2, axis=[0, 2, 3])
            yuv_scales = torch.Tensor(yuv_scales)
            if args.cuda:
                yuv_scales = yuv_scales.cuda()
            image_loss_yuv = torch.dot(yuv_loss, yuv_scales) # L2 norm for image similarity loss
            rgb_image_loss = l1_loss(stamp_perturbation, torch.zeros_like(stamp_perturbation))
            image_loss = image_loss_yuv + rgb_image_loss

            weight_img = max(0, min(args.img_w*(global_step-args.img_start)/(args.img_end-args.img_start), 1))

            loss_all = weight_img*(args.img_w*image_loss + args.lps_w*lpips_loss) + args.rs_w*loss_acc 
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            if global_step % args.log_interval == 0:
                name_space = ['image_loss:' , 'lpips_loss:' , 'loss_acc:',  'rgb_image_loss:']
                losses_log = [image_loss.item(), lpips_loss.item(),  loss_acc.item(),  rgb_image_loss.item()]
                print('Global_step {}: {}'.format(global_step, ['{}{}'.format(name_space[i],losses_log[i]) for i in range(len(losses_log))]))
                (ori_nonstamp_acc, ori_stamp_acc, ori_acc), (mix_nonstamp_acc, mix_stamp_acc, mix_acc), (apgd_nonstamp_acc, apgd_stamp_acc, apgd_acc) = validate(encoder, detector, val_loader, args)
                name_space = ['ori_nonstamp_acc', 'ori_stamp_acc', 'ori_acc', 'mix_nonstamp_acc', 'mix_stamp_acc', 'mix_acc', 'apgd_nonstamp_acc', 'apgd_stamp_acc', 'apgd_acc']
                val_accs = [ori_nonstamp_acc, ori_stamp_acc, ori_acc, mix_nonstamp_acc, mix_stamp_acc, mix_acc, apgd_nonstamp_acc, apgd_stamp_acc, apgd_acc]
                print('Evaluations {}: {}'.format(global_step, ['{}{}'.format(name_space[i],val_accs[i]) for i in range(len(val_accs))]))
                writer.add_scalar('val/ori_nonstamp_acc', ori_nonstamp_acc, global_step)
                writer.add_scalar('val/ori_stamp_acc', ori_stamp_acc, global_step)
                writer.add_scalar('val/ori_acc', ori_acc, global_step)
                writer.add_scalar('val/mix_nonstamp_acc', mix_nonstamp_acc, global_step)
                writer.add_scalar('val/mix_stamp_acc', mix_stamp_acc, global_step)
                writer.add_scalar('val/mix_acc', mix_acc, global_step)
                writer.add_scalar('val/apgd_nonstamp_acc', apgd_nonstamp_acc, global_step)
                writer.add_scalar('val/apgd_stamp_acc', apgd_stamp_acc, global_step)
                writer.add_scalar('val/apgd_acc', apgd_acc, global_step)
            if global_step % args.save_interval == 0:
                torch.save(encoder,  os.path.join(args.saved_models,  "encoder.pth"))
                torch.save(detector, os.path.join(args.saved_models, "detector.pth"))
            
            # write data into tensorboard
            if global_step % 20 == 0:
                lr = scheduler.get_last_lr()
                writer.add_scalar('loss/lr', lr[-1], global_step)
                writer.add_scalar('loss/loss_all', loss_all, global_step)
                writer.add_scalar('loss/image_loss', image_loss, global_step)
                writer.add_scalar('loss/lpips_loss', lpips_loss, global_step)
                writer.add_scalar('loss/loss_acc', loss_acc, global_step)
                writer.add_scalar('loss/rgb_image_loss', rgb_image_loss, global_step)
                writer.add_scalar('loss/image_loss_yuv', image_loss_yuv, global_step)
            if global_step % 500 == 0:
                writer.add_image('input/image_input', image_input[0], global_step)
                writer.add_image('encoded/stamp_perturbation', stamp_perturbation[0]+0.5, global_step)
                writer.add_image('encoded/stamped_image', stamped_image[0], global_step)
            global_step += 1
            
        if global_step>=args.max_step:
            break
        scheduler.step()

    writer.close()
    torch.save(encoder, os.path.join(args.saved_models, "encoder.pth"))
    torch.save(detector, os.path.join(args.saved_models, "detector.pth"))
    # Evaluations
    evaluate(encoder, detector, test_loader, args)

def validate(encoder, detector, val_loader, args):
    encoder.eval()
    detector.eval()
    ori_nonstamp_acc = 0
    ori_stamp_acc = 0
    ori_acc = 0 
    mix_nonstamp_acc = 0
    mix_stamp_acc    = 0
    mix_acc          = 0
    apgd_nonstamp_acc = 0
    apgd_stamp_acc    = 0
    apgd_acc          = 0
    scale_1 = np.random.uniform(0, 0.4)
    degree_scale = np.random.randint(0,45)
    cropper = T.RandomResizedCrop(size=(400, 400), scale=(scale_1, 1))
    rotater = T.RandomRotation(degrees=(-degree_scale, degree_scale))
    jitter = T.ColorJitter(brightness=scale_1, contrast=scale_1, saturation=scale_1, hue=scale_1)
    blurrer = T.GaussianBlur(kernel_size=(3, 7), sigma=(1, 3))
    perspective_transformer = T.RandomPerspective(distortion_scale=scale_1, p=1.0)
    jpeg = DiffJPEG(height=400, width=400, differentiable=True, quality=int(np.random.uniform(50,100,1)))
    jpeg = jpeg.cuda()
    
    zeros  = torch.zeros(args.batch_size,  1, requires_grad=False, dtype = int).cuda()
    ones   = torch.ones (args.batch_size,  1, requires_grad=False, dtype = int).cuda()
    with torch.no_grad():
        for batch_idx, image_input in enumerate(val_loader, 0):
            image_input = image_input.cuda() 
            stamp_perturbation = encoder(image_input)
            stamped_image = image_input + stamp_perturbation
            
            img_non_stmp = image_input
            img_stmp = stamped_image

            non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
            ori_nonstamp_acc += non_stmp_acc
            ori_stamp_acc    += stmp_acc
            ori_acc          += (non_stmp_acc + stmp_acc)/2

            new_image = image_input
            if np.random.rand()<=1: new_image = jitter(new_image)
            if np.random.rand()<=1: new_image = cropper(new_image)
            if np.random.rand()<=1: new_image = blurrer(new_image)
            if np.random.rand()<=1: new_image = rotater(new_image)
            if np.random.rand()<=1: new_image = perspective_transformer(new_image)
            if np.random.rand()<=1: new_image = jpeg(new_image)
            img_non_stmp = new_image

            new_image = stamped_image
            if np.random.rand()<=1: new_image = jitter(new_image)
            if np.random.rand()<=1: new_image = cropper(new_image)
            if np.random.rand()<=1: new_image = blurrer(new_image)
            if np.random.rand()<=1: new_image = rotater(new_image)
            if np.random.rand()<=1: new_image = perspective_transformer(new_image)
            if np.random.rand()<=1: new_image = jpeg(new_image)
            img_stmp = new_image

            non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp, img_stmp, ones, zeros, args.std)
            mix_nonstamp_acc += non_stmp_acc
            mix_stamp_acc    += stmp_acc
            mix_acc          += (non_stmp_acc + stmp_acc)/2
    ori_nonstamp_acc = 100*ori_nonstamp_acc/(batch_idx+1)
    ori_stamp_acc    = 100*ori_stamp_acc/(batch_idx+1)
    ori_acc          = 100*ori_acc/(batch_idx+1)
    mix_nonstamp_acc = 100*mix_nonstamp_acc/(batch_idx+1)
    mix_stamp_acc    = 100*mix_stamp_acc/(batch_idx+1)
    mix_acc          = 100*mix_acc/(batch_idx+1) 
    
    ## adverairal robustness evaluation: pgd-ce attack
    with torch.no_grad():
        ### auto attack [square + apg-ce]
        auto_attack = AutoAttack(detector, norm='Linf', eps=8/255, version='standard')
        auto_attack.attacks_to_run = [ 'apgd-ce']
        image_inputs = [image_input for (image_input ) in val_loader]
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
            zeros_lb  = torch.zeros(img_non_stmp_adv.size()[0], 1,  requires_grad=False, dtype = int)
            ones_lb   = torch.ones (img_non_stmp_adv.size()[0], 1,  requires_grad=False, dtype = int)
            non_stmp_acc, stmp_acc = acc_calculate(detector, img_non_stmp_adv.cuda(), img_stmp_adv.cuda(), ones_lb.cuda(), zeros_lb.cuda(), args.std)
            apgd_nonstamp_acc += non_stmp_acc
            apgd_stamp_acc    += stmp_acc
            apgd_acc          += (non_stmp_acc + stmp_acc)/2

        apgd_nonstamp_acc = 100*apgd_nonstamp_acc/(i1+1)
        apgd_stamp_acc    = 100*apgd_stamp_acc/(i1+1)
        apgd_acc          = 100*apgd_acc/(i1+1)
        print(ori_nonstamp_acc, ori_stamp_acc, ori_acc, mix_nonstamp_acc, mix_stamp_acc, mix_acc, apgd_nonstamp_acc, apgd_stamp_acc, apgd_acc)
    
    encoder.train()
    detector.train()
    return (ori_nonstamp_acc, ori_stamp_acc, ori_acc), (mix_nonstamp_acc, mix_stamp_acc, mix_acc), (apgd_nonstamp_acc, apgd_stamp_acc, apgd_acc)
    
    
if __name__ == '__main__':
    dataset_folder = './mirflickr/'
    if not os.path.exists(dataset_folder):
        os.system('wget http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip')
        os.system('unzip mirflickr25k.zip')
        os.system('rm -f mirflickr25k.zip')

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=2000)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--tensorboard', type=int, default=1)
    parser.add_argument('--l2_loss_scale', type=float, default=2)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--rs_w', type=float, default=1)
    parser.add_argument('--rtt', type=float, default=1)
    parser.add_argument('--jtr', type=float, default=1)
    parser.add_argument('--cc', type=float, default=1)
    parser.add_argument('--blr', type=float, default=1)
    parser.add_argument('--wrp', type=float, default=1)
    parser.add_argument('--std', type=float, default=0.5)
    parser.add_argument('--reg_start', type=int, default=0)
    parser.add_argument('--img_w', type=float, default=1)
    parser.add_argument('--img_start', type=int, default=0)
    parser.add_argument('--img_end', type=int, default=1)
    parser.add_argument('--reg_w', type=float, default=1)
    parser.add_argument('--lps_w', type=float, default=1)
    parser.add_argument('--y_scale', type=int, default=1)
    parser.add_argument('--u_scale', type=int, default=100)
    parser.add_argument('--v_scale', type=int, default=100)
    parser.add_argument('--max_step', type=int, default=40000)
    parser.add_argument('--rtt_set', type=float, default=30)
    parser.add_argument('--wrp_set', type=float, default=0.4)
    parser.add_argument('--crp_set', type=float, default=0.4)
    parser.add_argument('--ptb_start', type=int, default=50000)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--detector', type=str, default='iv3')
    parser.add_argument('--aug_prb', type=float, default=0.75)
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--crp_dropout', type=float, default=0.9)
    user_args = parser.parse_args()
    main(vars(user_args))
