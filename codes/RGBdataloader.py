import glob
import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from imageio import imread

class RGBDataset(Dataset):
    def get_data_dir(self,root_dir):
        rgb_file_path = []
        depth_file_path = []
        gt_file_path = []
        dirs1 = os.listdir(root_dir)
        for file1 in dirs1:
            file_path = os.path.join(root_dir,file1)

            if self.train:
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

        return rgb_file_path,depth_file_path,gt_file_path
    def get_data(self):
            rgb_file_path,depth_file_path,gt_file_path =self.get_data_dir(self.root_dir)
            rgb_file = []
            depth_file = []
            gt_file = []
            for file in rgb_file_path:
                img =  cv2.imread(file[0])
                #img = np.transpose(img,(2,0,1))
                rgb_file.append(img)
            rgb_file = np.array(rgb_file,dtype = float)
            #rgb_file = np.transpose(rgb_file,(0,3,1,2))

            for file in depth_file_path:
                img = imread(file[0])
                depth_file.append(img)
            depth_file = np.array(depth_file,dtype = float)
           
            for file in gt_file_path:
                img = imread(file[0])
                
                gt_file.append(img)
            gt_file = np.array(gt_file,dtype = float)


            return rgb_file,depth_file,gt_file
    def minmax (self,a):
        minn = np.min(a)
        maxx = np.max(a)
        a = (a-minn) / (maxx - minn)
        return a
    def __init__(self,root_dir,transform = None,train = True,scale = 4):
        self.root_dir = root_dir 
        self.transform = transform
        self.scale = scale
        self.train = train
        self.rgb_file,self.depth_file,self.gt_file = self.get_data()
    
    def __getitem__(self,idx):
        
        image = self.rgb_file[idx]/255.0
        
        depth = self.depth_file[idx]
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        depth = self.minmax(depth)
        gt = self.gt_file[idx]
      
        if self.train:
            gt = self.minmax(gt)
        
        h,w = gt.shape
        s = self.scale
        target = np.array(Image.fromarray(depth).resize((w//s,h//s),Image.BICUBIC).resize((w, h), Image.BICUBIC))
        
        if self.transform:
            image = self.transform(image).float()
            gt = self.transform(np.expand_dims(gt,2)).float()
            target = self.transform(np.expand_dims(target,2)).float()
        sample = {'guidance': image, 'target': target, 'gt': gt,'depth_min':depth_min,'depth_max':depth_max}
        
        return sample

    def __len__(self):
        return self.depth_file.shape[0]

    
        
