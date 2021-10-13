import  os 
import imageio 
import glob
import cv2
import numpy as np
from torchvision import transforms, utils

train = True
root_dir = "/data/yonghui/RGB-D-D"

rgb_file_path = []
depth_file_path = []
gt_file_path = []

dirs1 = os.listdir(root_dir)
for file1 in dirs1:
    file_path=os.path.join(root_dir,file1)
    if train:
        file_path = os.path.join(file_path,"*train") 
    else:
        file_path = os.path.join(file_path,"*test")

    file_path = glob.glob(file_path)
    dirs2 = os.listdir(file_path[0])
    for file2 in dirs2:
        rgb_path = os.path.join(file_path[0],file2)
        rgb_path = os.path.join(rgb_path,"*.jpg")
        rgb_path = glob.glob(rgb_path)
        rgb_file_path.append(rgb_path)
                    
        depth_path = os.path.join(file_path[0],file2)
        depth_path = os.path.join(depth_path,"*LR_fill_depth.png")
        depth_path = glob.glob(depth_path)
        depth_file_path.append(depth_path)

        gt_path = os.path.join(file_path[0],file2)
        gt_path = os.path.join(gt_path,"*gt.png")
        gt_path = glob.glob(gt_path)
        gt_file_path.append(gt_path)

rgb_file = []
for file in depth_file_path:
    img =  cv2.imread(file[0],0).astype('float32')
    rgb_file.append(img)
rgb_file = np.array(rgb_file)
#rgb_file = np.transpose(rgb_file,(0,3,1,2))

img = rgb_file[0]

transform = transforms.Compose([transforms.ToTensor()])
img = transform(img).float()
print(1)