import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import cv2
import argparse
import time
import glob
from imageio import imread
from models import *
from PIL import Image
from DNN_printer import DNN_printer

parser = argparse.ArgumentParser()
parser.add_argument('--rgb_files',  default='./RGB-D-D-Dataset-main/data/RGB_20/*.jpg', help='folder name of rgb image')
parser.add_argument('--depth_files',  default='./RGB-D-D-Dataset-main/data/LR_FILL_20/*.png', help='folder name of low resolution depth image')
parser.add_argument('--scale', type=int, default=4, help='scale factor')

parser.add_argument('--parameter',  default='./RGB-D-D-Dataset-main/data/parameter/RMSE 5.4', help='name of parameter file')
parser.add_argument('--model',  default='FDSR', help='choose model')
parser.add_argument('--output_folder',  default='/data1/users/yonghui/RGB-D-D/RGB-D-D-Dataset-main/data/out_put/', help='name of output image')
opt = parser.parse_args()
print(opt)


def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_trainable_params



def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo

    return image[:h,:w]

test_files_down = glob.glob(opt.depth_files)
test_files_down = sorted(test_files_down)
test_files_rgb = glob.glob(opt.rgb_files)
test_files_rgb = sorted(test_files_rgb)
print(test_files_rgb)
print(test_files_down)





net = Net(num_feats=32, depth_chanels=1, color_channel=3, kernel_size=3).cuda()
net = nn.DataParallel(net)
net.load_state_dict(torch.load(opt.parameter))
   
net.eval()

cnn_paras_count(net)
for i in range (len(test_files_down)):   
   
    rgb = cv2.imread(test_files_rgb[i]).astype('float32') / 255.0  #cv2.imread 读取出的信息排列方式为   图片的H，W，C
    rgb = np.transpose(rgb, (2, 0, 1))   #将其转换为C，H，W
    lr = imread(test_files_down[i]).astype('float32')    #lr:低分辨率图像

    lr = np.array(Image.fromarray(lr).resize((128,96),Image.BICUBIC).resize((512,384), Image.BICUBIC))
    #lr = np.array(Image.fromarray(lr).resize((160,120),Image.BICUBIC).resize((640,480), Image.BICUBIC))

    maxx=np.max(lr)
    minn=np.min(lr)
    lr_o=(lr-minn)/(maxx-minn)
    lr = np.expand_dims(lr_o, 0)
    image = torch.from_numpy(np.expand_dims(rgb, 0)).cuda()
    depth = torch.from_numpy(np.expand_dims(lr, 0)).cuda()
    t4 = time.time()
    with torch.no_grad():
        res_img = net((image, depth)).cpu().numpy()
    t5 = time.time()
    count_wa = 0
    h, w = res_img[0,0].shape[0], res_img[0,0].shape[1]
    print("h,w",h,w)

    output = (res_img[0, 0]*(maxx-minn)+minn).astype(np.uint16)
    output2 = output
    tx = time.time()
    temp = output > 60000
    
    output[temp] = lr_o[temp]
    ty = time.time()    
    
   
    print("t",ty-tx)
    cv2.imwrite(opt.output_folder  + test_files_down[i][39:], output)
    t6 = time.time()
    
    
   


#dirs = opt.output_folder
#output=glob.glob(dirs+'*.png')
#output = sorted(output)



#dir1GT='./data/gt/'
#dirshuafen= opt.output_folder
#GT=glob.glob(dir1GT+'*.png')
#GT = sorted(GT)
#output=glob.glob(dirshuafen+'*.png')
#output = sorted(output)
#rmse=0
#print(len(GT),len(output))
#for i in range(len(output)):
#    gg=imread(GT[i]).astype(np.float32)     #读取GT  并将其转化为float类型 
#    oo=imread(output[i]).astype(np.float32)
#    
#   gg = gg[6:-6, 6:-6]
#    oo = oo[6:-6, 6:-6]
#    gg = gg/10.0
#    oo = oo/10.0

#    res=np.sqrt(np.mean(np.power(gg - oo, 2)))
#    rmse+=res
#    print(GT[i],output[i],res)
#print(rmse/len(output))
