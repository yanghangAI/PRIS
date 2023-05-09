#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import calculate_PSNR_SSIM
import numpy as np
from model import *
import config as c
from torch.utils.tensorboard import SummaryWriter
import datasets
import viz
import warnings
from util import attack, gauss_noise, mse_loss, computePSNR, dwt, iwt
import torchvision

# 网络参数数量

def load(net, optim, name, load_opt=True):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    if load_opt == True:
        optim.load_state_dict(state_dicts['opt'])
    return net, optim


def embed_attack(net, input_img, attack_method):
    #################
    #    forward:   #
    #################
    output = net(input_img)
    output_steg = output.narrow(1, 0, 4 * c.channels_in)
    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
    output_z = gauss_noise(output_z.shape)
    steg_img = iwt(output_steg)

    #################
    #   attack:   #
    #################
    attack_steg = attack(steg_img, attack_method)
    input_steg = dwt(attack_steg)

    return steg_img, attack_steg, output_z, output_steg, input_steg

def train_epoch(net, step, optim=None, attack_method=None, i_epoch=None, writer=None, mode='train', lam=(1.0, 1.0), device='cuda'):
    r_loss_list, g_loss_list, pre_loss_list, post_loss_list, psnr_c, psnr_s, total_loss_list = [], [], [], [], [], [], []
    lam_c, lam_s = lam
    if mode != 'train':
        dataloader = datasets.testloader
        net.eval()
    else:
        dataloader = datasets.trainloader
        net.train()

    for i_batch, data in enumerate(dataloader):
        data = data.to(device)
        num = data.shape[0] // 2
        cover = data[:num]
        secret = data[num:num * 2]
        cover_input = dwt(cover)
        secret_input = dwt(secret)

        input_img = torch.cat((cover_input, secret_input), 1)
        steg_img, attack_steg, output_z, output_steg, input_steg = embed_attack(net, input_img, attack_method)

        if step == 1 or step == 2:
            input_steg = net.pre_enhance(attack_steg)
            input_steg = dwt(input_steg)

        #################
        #   backward:   #
        #################
        output_rev = torch.cat((input_steg, output_z), 1)
        output_image = net(output_rev, rev=True)
        secret_rev = output_image.narrow(1, 4 * c.channels_in,
                                         output_image.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        if step == 1 or step == 2:
            secret_rev = net.post_enhance(secret_rev)


        #################
        #     loss:     #
        #################

        c_loss = mse_loss(steg_img, cover)
        s_loss = mse_loss(secret_rev, secret)

        secret_rev = secret_rev.clip(0, 1)
        secret = secret.clip(0, 1)
        cover = cover.clip(0, 1)
        steg = steg_img.clip(0, 1)

        psnr_temp = computePSNR(secret_rev, secret)
        psnr_s.append(psnr_temp)
        psnr_temp_c = computePSNR(cover, steg)
        psnr_c.append(psnr_temp_c)

        if step == 1:
            total_loss = s_loss
        elif step == 0 or step == 2:
            total_loss = lam_s * s_loss + lam_c * c_loss


        if mode == 'train':
            total_loss.backward()
            optim.step()
            optim.zero_grad()

        elif mode == 'test':
            torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.png' % i_batch)
            torchvision.utils.save_image(steg, c.IMAGE_PATH_steg + '%.5d.png' % i_batch)
            torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.png' % i_batch)
            torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.png' % i_batch)

        g_loss_list.append(c_loss.item())
        r_loss_list.append(s_loss.item())
        total_loss_list.append(total_loss.item())

    if mode == 'val':
        before = 'val_'
    else:
        before = ''

    if mode != 'test':
        writer.add_scalars(f"{before}c_loss", {f"{before}guide loss": np.mean(g_loss_list)}, i_epoch)
        writer.add_scalars(f"{before}s_loss", {f"{before}rev loss": np.mean(r_loss_list)}, i_epoch)
        writer.add_scalars(f"{before}PSNR_S", {f"{before}average psnr": np.mean(psnr_s)}, i_epoch)
        writer.add_scalars(f"{before}PSNR_C", {f"{before}average psnr": np.mean(psnr_c)}, i_epoch)
        writer.add_scalars(f"{before}Loss", {f"{before}Loss": np.mean(total_loss_list)}, i_epoch)

    return np.mean(total_loss_list)



def train(net, step, optim, weight_scheduler, attack_method, start_epoch, end_epoch, visualizer=None, expinfo='', lam=(1.0, 1.0)):

    writer = SummaryWriter(comment=expinfo, filename_suffix="steg")
    val_loss = train_epoch(net, step, optim, attack_method, start_epoch, mode='val', writer=writer, lam=lam)

    for i_epoch in range(start_epoch + 1, end_epoch + 1):

        #################
        #     train:    #
        #################
        train_loss = train_epoch(net, step, optim, attack_method, i_epoch, writer=writer, lam=lam)

        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0:
            with torch.no_grad():
                val_loss = train_epoch(net, step, optim, attack_method, i_epoch, mode='val', writer=writer, lam=lam)

        info = [np.round(train_loss, 2), np.round(val_loss, 2), np.round(np.log10(optim.param_groups[0]['lr']), 2), attack_method]

        viz.show_loss(visualizer, info)

        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, c.MODEL_PATH + 'model_checkpoint_%.5i' % i_epoch + '.pt')

        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, f'final_state/{expinfo}.pt')
    writer.close()


def model_init(step, load_path='', load_opt=True):
    net = PRIS(in_1=3, in_2=3)
    if step == 0:
        lr = c.lr
        for name, para in net.named_parameters():
            if 'inbs' in name:
                para.requires_grad = True
            elif 'enhance' in name:
                para.requires_grad = False

    elif step == 1:
        lr = c.lr
        for name, para in net.named_parameters():
            if 'inbs' in name:
                para.requires_grad = False
            elif 'enhance' in name:
                para.requires_grad = True
    elif step == 2:
        lr = c.lr * 0.1
        for name, para in net.named_parameters():
            if 'inbs' in name:
                para.requires_grad = True
            elif 'enhance' in name:
                para.requires_grad = True


    optim = torch.optim.Adam(filter(lambda x:x.requires_grad, net.parameters()), lr=lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    net.cuda()
    init_model(net)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    if load_path != '':
        net, optim = load(net, optim, load_path, load_opt)
        print(f'load from {load_path}')
    return net, optim, weight_scheduler



def main(attack_method, step, load_path='', start_epoch=0, end_epoch = 1600, lam=(1.0, 1.0)):
    warnings.filterwarnings("ignore")
    if step == 0:
        expinfo = f'{attack_method}_hinet_pretrain'
    elif step == 1:
        expinfo = f'{attack_method}_enhance_pretrain'
    elif step == 2:
        expinfo = f'{attack_method}_enhance_finetune'


    if load_path == '':
        load_opt = False
        if step == 1:
            load_path = f'final_state/{attack_method}_hinet_pretrain.pt'
        elif step == 2:
            load_path = f'final_state/{attack_method}_enhance_pretrain.pt'
    else:
        load_opt = True


    net, optim, weight_scheduler = model_init(step=step, load_path=load_path, load_opt=load_opt)

    visualizer = viz.Visualizer(c.loss_names)
    train(net, step, optim, weight_scheduler, attack_method, start_epoch, end_epoch, visualizer, expinfo=expinfo, lam=lam)
    train_epoch(net, attack_method=attack_method, mode='test', step=step)
    calculate_PSNR_SSIM.main(f'{expinfo}')



if __name__ == '__main__':
    attack_method = 'gaussian10'
    lambda_c = 1.0
    lambda_s = 1.0
    lam = (lambda_c, lambda_s)
    for step in range(3):
        main(attack_method, step, start_epoch=0, end_epoch=1600, lam=lam)




