import os
import os.path
import cv2
import numpy as np
import copy

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

from .get_weak_anns import transform_anns

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None, filter_intersection=False):    
    assert split in [0, 1, 2, 3]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)  
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []     

        if filter_intersection:        
            if set(label_class).issubset(set(sub_list)):
                for c in label_class:
                    if c in sub_list:
                        tmp_label = np.zeros_like(label)
                        target_pix = np.where(label == c)  
                        tmp_label[target_pix[0],target_pix[1]] = 1 
                        if tmp_label.sum() >= 2 * 32 * 32:      
                            new_label_class.append(c)     
        else:
            for c in label_class:      
                if c in sub_list:
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c) 
                    tmp_label[target_pix[0],target_pix[1]] = 1 
                    if tmp_label.sum() >= 2 * 32 * 32:      
                        new_label_class.append(c)            

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)  
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)  
                    
    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list



class SemData(Dataset):
    def __init__(self, split=0, shot=1, data_root=None, data_list=None, data_set=None, use_split_coco=False, \
                        transform=None, mode='train', ann_type='mask', \
                        ft_transform=None, ft_aug_size=None, \
                        ms_transform=None):

        assert mode in ['train', 'val', 'demo', 'finetune']
        assert data_set in ['pascal', 'coco']
        if mode == 'finetune':
            assert ft_transform is not None
            assert ft_aug_size is not None

        self.mode = mode
        self.split = split  
        self.shot = shot
        self.data_root = data_root   
        self.ann_type = ann_type

        if data_set == 'pascal':
            self.class_list = list(range(1, 21))                         # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

            if self.split == 3: 
                self.sub_list = list(range(1, 16))                       # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))                  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))                  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))                   # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))                       # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))                    # [1,2,3,4,5]

        elif data_set == 'coco':
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))           

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    


        fss_list_root = './lists/{}/fss_list/{}/'.format(data_set, self.mode)
        fss_data_list_path = fss_list_root + 'data_list_{}.txt'.format(split)
        fss_sub_class_file_list_path = fss_list_root + 'sub_class_file_list_{}.txt'.format(split)

        with open(fss_data_list_path, 'r') as f:
            f_str = f.readlines()
        self.data_list = []
        for line in f_str:
            img, mask = line.split(' ')
            self.data_list.append((img, mask.strip()))

        with open(fss_sub_class_file_list_path, 'r') as f:
            f_str = f.read()
        self.sub_class_file_list = eval(f_str)

        self.transform = transform
        self.ft_transform = ft_transform
        self.ft_aug_size = ft_aug_size
        self.ms_transform_list = ms_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]  
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)


        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))          
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255) 
        new_label_class = []       
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'demo' or self.mode == 'finetune':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class    
        assert len(label_class) > 0

        class_chosen = label_class[random.randint(1,len(label_class))-1]  
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0],target_pix[1]] = 1 
        label[ignore_pix[0],ignore_pix[1]] = 255     


        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):  
            support_idx = random.randint(1,num_file)-1
            support_image_path = image_path
            support_label_path = label_path
            while((support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):  
                support_idx = random.randint(1,num_file)-1
                support_image_path, support_label_path = file_class_chosen[support_idx]                
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list_ori = []
        support_label_list_ori = []      
        support_label_list_ori_mask = [] 


        subcls_list = []
        for k in range(self.shot):  
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))

            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k] 
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)      
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)
            support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:,:] = 0
            support_label[target_pix[0],target_pix[1]] = 1 
            
            support_label, support_label_mask = transform_anns(support_label, self.ann_type)  
            support_label[ignore_pix[0],ignore_pix[1]] = 255
            support_label_mask[ignore_pix[0],ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError("Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))            
            support_image_list_ori.append(support_image)
            support_label_list_ori.append(support_label)
            support_label_list_ori_mask.append(support_label_mask)
        assert len(support_label_list_ori) == self.shot and len(support_image_list_ori) == self.shot    



        raw_image = image.copy()
        raw_label = label.copy()
        support_image_list = [[] for _ in range(self.shot)]
        support_label_list = [[] for _ in range(self.shot)]

        if self.transform is not None:
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list_ori[k], support_label_list_ori[k])
       

        s_xs = support_image_list
        s_ys = support_label_list
        
        #s_cs = torch.Tensor(support_classlabel_list)
        #u_cs = torch.Tensor(unlabel_classlabel_list)
   
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

   
        # Finetune
        if self.mode == 'finetune':
            num_aug_per_img = self.ft_aug_size
            support_image_aug_list = []
            support_label_aug_list = []
            for k in range(self.shot):
                for a_id in range(num_aug_per_img):      
                    image_aug_temp, label_aug_temp = self.ft_transform(support_image_list_ori[k], support_label_list_ori[k])
                    support_image_aug_list.append(image_aug_temp)
                    support_label_aug_list.append(label_aug_temp)

            assert len(support_image_aug_list) == self.shot * num_aug_per_img
            s_x_aug = support_image_aug_list[0].unsqueeze(0)
            for i in range(1, len(support_image_aug_list)):
                s_x_aug = torch.cat([support_image_aug_list[i].unsqueeze(0), s_x_aug], 0)
            s_y_aug = support_label_aug_list[0].unsqueeze(0)
            for i in range(1, len(support_image_aug_list)):
                s_y_aug = torch.cat([support_label_aug_list[i].unsqueeze(0), s_y_aug], 0)

        # Multi-Scale
        if self.ms_transform_list is not None:
            image_list = []
            label_list = []
            support_image_list = []
            support_label_list = []
            for ms_id in range(len(self.ms_transform_list)):
                ms_transform_temp = self.ms_transform_list[ms_id]
                scale_img, scale_label = ms_transform_temp(raw_image, raw_label)

                scale_img_s, scale_label_s = ms_transform_temp(support_image_list_ori[0], support_label_list_ori[0])
                s_x = scale_img_s.unsqueeze(0)
                s_y = scale_label_s.unsqueeze(0)
                for k in range(1, self.shot):
                    scale_img_s, scale_label_s = ms_transform_temp(support_image_list_ori[k], support_label_list_ori[k])
                    s_x = torch.cat([scale_img_s.unsqueeze(0), s_x], 0)
                    s_y = torch.cat([scale_label_s.unsqueeze(0), s_y], 0)

                image_list.append(scale_img)
                label_list.append(scale_label)
                support_image_list.append(s_x)
                support_label_list.append(s_y)
            image = image_list
            label = label_list
            s_x = support_image_list
            s_y = support_label_list
        
        total_image_list = support_image_list_ori.copy()
        total_image_list.append(raw_image)
        # Return
        if self.mode == 'train':
            return image, label, s_x, s_y, subcls_list
        elif self.mode == 'val':
            return image, label, s_x, s_y, subcls_list, raw_label
        elif self.mode == 'demo':
            return image, label, s_x, s_y, subcls_list, total_image_list, support_label_list_ori, support_label_list_ori_mask, raw_label            
        elif self.mode == 'finetune':
            return image, label, s_x, s_y, s_x_aug, s_y_aug, subcls_list, raw_label
