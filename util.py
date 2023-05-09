import os
import logging
import random
from datetime import datetime
import torch
from DiffJPEG import DiffJPEG
from modules.Unet_common import IWT, DWT


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def round_diff(x):
    sign = torch.ones_like(x)
    sign[torch.floor(x) % 2 == 0] = -1
    y = sign * torch.cos(x * torch.pi) / 2
    out = torch.round(x) + y - y.detach()
    return out

def attack(img, method):

    if method[:8] == 'gaussian':
        level = int(method[8:])
        img = img + level * torch.randn(img.shape).to(img.device) / 255.
        img = img.clip(0, 1)


    elif method[:7] == 'JPEG Q=':
        level = int(method[7:])
        size = img.shape
        img = img * 255
        jpeg = DiffJPEG(size[-1], size[-2], differentiable=True, quality=level).to(img.device)
        img = jpeg(img)
        img = img / 255

    elif method == 'round':
        img = img * 255
        img = round_diff(img)
        img = img / 255


    elif method == 'none':
        pass

    elif method == 'mix':
        img = attack(img, 'round')
        rand = random.choice(['gaussian1', 'gaussian10', 'JPEG Q=90', 'JPEG Q=80'])
        img = attack(img, rand)

    elif method == 'mix2':
        rand = random.choice(['gaussian1', 'gaussian10', 'JPEG Q=90', 'JPEG Q=80', 'round', 'gaussian10', 'JPEG Q=80'])
        img = attack(img, rand)

    else:
        print('no attack is taken')

    return img

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise

def mse_loss(a, b):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(a, b)
    return loss

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def computePSNR(origin,pred):
    mse = torch.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * torch.log10(1.0/mse).item()

iwt = IWT()
dwt = DWT()
