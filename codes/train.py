import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import torch
import numpy as np
import cv2
import argparse
from RGBdataloader import *
from models import *
from nyu_dataloader import *

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=4, help='scale factor')
parser.add_argument('--parameter',  default='/data/yonghui/NYU/parameter/', help='name of parameter file')
parser.add_argument('--model',  default='FDSR', help='choose model')
parser.add_argument('--lr',  default='0.0005', type=float, help='learning rate')
parser.add_argument('--result',  default='/data/yonghui/NYU/result/', help='learning rate')
parser.add_argument('--epoch',  default=50, type=int, help='max epoch')

def calc_rmse(a, b,depth_min,depth_max):
    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    b = b * (depth_max-depth_min) + depth_min
    a = a/10.0
    b = b/10.0
    #a = a*(minmax[1]-minmax[0]) + minmax[1]
    #b = b*(minmax[1]-minmax[0]) + minmax[1]
    
    return np.sqrt(np.mean(np.power(a-b,2)))

@torch.no_grad()

def validate(net, root_dir='/data/yonghui/NYU'):

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    #test_dataset = NYU_v2_datset(root_dir=root_dir, transform=data_transform, train=False)
    test_dataset = RGBDataset(root_dir = "/data/yonghui/RGB-D-D",transform=data_transform,train = False)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    net.eval()
    rmse = np.zeros(405)
    #test_minmax = np.load('%s/test_minmax.npy'%root_dir)
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        #minmax = test_minmax[:,idx]
        
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
        depth_min ,depth_max= data['depth_min'].numpy()[0],data['depth_max'].numpy()[0]
        out = net((guidance, target))
        rmse[idx] = calc_rmse(gt[0,0].cpu().numpy(), out[0,0].cpu().numpy(),depth_min,depth_max)
        
        t.set_description('[validate] rmse: %f' %rmse[:idx+1].mean())
        t.refresh()
    
    return rmse





opt = parser.parse_args()
print(opt)

s = datetime.now().strftime('%Y%m%d%H%M%S')
result_root = '%s%slr_%s-s_%s'%(opt.result, s, opt.lr, opt.scale)
if not os.path.exists(result_root): 
    os.makedirs(result_root)

logging.basicConfig(filename='%s/train.log'%result_root,format='%(asctime)s %(message)s', level=logging.INFO)

net = Net(num_feats=32, depth_chanels=1, color_channel=3, kernel_size=3).cuda()
net = nn.DataParallel(net)
criterion = nn.L1Loss()  
optimizer = optim.Adam(net.parameters(), lr=opt.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=40000, gamma=0.5)
net.train()

data_transform = transforms.Compose([transforms.ToTensor()])

#nyu_dataset = NYU_v2_datset(root_dir='/data/yonghui/NYU', transform=data_transform)
RGB_dataset = RGBDataset(root_dir = "/data/yonghui/RGB-D-D",transform=data_transform)

dataloader = torch.utils.data.DataLoader(RGB_dataset, batch_size=1, shuffle=True)

max_epoch = opt.epoch
for epoch in range(max_epoch):
    net.train()
    running_loss = 0.0
    
    t = tqdm(iter(dataloader), leave=True, total=len(dataloader))
    for idx, data in enumerate(t):
        optimizer.zero_grad()
        scheduler.step()
        guidance, target, gt = data['guidance'].cuda(), data['target'].cuda(), data['gt'].cuda()
      
        #print(guidance)
        out = net((guidance, target))
        
        loss = criterion(out, gt)
        loss.backward()
        optimizer.step()
        running_loss += loss.data.item()
        
        if idx % 25 == 0:
            if idx != 0:
                running_loss /= 25
            t.set_description('[train epoch(L1):%d] loss: %.10f' % (epoch+1, running_loss))
            t.refresh()
            logging.info('epoch:%d running_loss:%.10f' % (epoch + 1, running_loss))
        #print(optimizer) 
    rmse = validate(net)

    logging.info('epoch:%d --------mean_rmse:%.10f '%(epoch+1, rmse.mean()))
    torch.save(net.state_dict(), "%s/parameter%d"%(result_root, epoch+1))