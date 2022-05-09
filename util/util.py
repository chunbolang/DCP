import os
import numpy as np
from PIL import Image
import random
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib import font_manager
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import math
from seaborn.distributions import distplot
from tqdm import tqdm
from scipy import ndimage

# from get_weak_anns import find_bbox, ScribblesRobot

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.init as initer


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False, warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power   # warmup时不连续

    if curr_iter % 50 == 0:   
        print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))     

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr   # 都是10倍学习率


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index      # 将GT中255的部分对应赋值给output
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)  # 排除掉255(0~K-1之间) 返回2个类别的元素个数(bins=2)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  # 取文件绝对路径
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):#, BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


# ------------------------------------------------------
def get_model_para_number(model):
    total_number = 0
    learnable_number = 0 
    for para in model.parameters():
        total_number += torch.numel(para)
        if para.requires_grad == True:
            learnable_number+= torch.numel(para)
    return total_number, learnable_number

def setup_seed(seed=2021, deterministic=False):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_metirc(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    #交并比
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index      # 将GT中255的部分对应赋值给output
    intersection = output[output == target]
    if intersection.shape[0] == 0:
        area_intersection = torch.tensor([0.,0.],device='cuda')
    else:
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)  # 排除掉255(0~K-1之间) 返回2个类别的元素个数(bins=2)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    # area_union = area_output + area_target - area_intersection
    Pre = area_intersection / (area_output + 1e-10)
    Rec = area_intersection / (area_target + 1e-10)
    return Pre, Rec

def get_save_path(args):

    if args.vgg:
        backbone_str = 'vgg'
    else:
        backbone_str = 'resnet' + str(args.layers)
    args.snapshot_path = 'exp/{}/{}/split{}/{}/snapshot'.format(args.data_set, args.arch, args.split, backbone_str)
    args.result_path = 'exp/{}/{}/split{}/{}/result'.format(args.data_set, args.arch, args.split, backbone_str)

def get_train_val_set(args):
    if args.data_set == 'pascal':
        class_list = list(range(1, 21)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        if args.split == 3: 
            sub_list = list(range(1, 16)) #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            sub_val_list = list(range(16, 21)) #[16,17,18,19,20]
        elif args.split == 2:
            sub_list = list(range(1, 11)) + list(range(16, 21)) #[1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
            sub_val_list = list(range(11, 16)) #[11,12,13,14,15]
        elif args.split == 1:
            sub_list = list(range(1, 6)) + list(range(11, 21)) #[1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(6, 11)) #[6,7,8,9,10]
        elif args.split == 0:
            sub_list = list(range(6, 21)) #[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            sub_val_list = list(range(1, 6)) #[1,2,3,4,5]

    elif args.data_set == 'coco':
        if args.use_split_coco:
            print('INFO: using SPLIT COCO (FWB)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_val_list = list(range(4, 81, 4))
                sub_list = list(set(class_list) - set(sub_val_list))                    
            elif args.split == 2:
                sub_val_list = list(range(3, 80, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
            elif args.split == 1:
                sub_val_list = list(range(2, 79, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
            elif args.split == 0:
                sub_val_list = list(range(1, 78, 4))
                sub_list = list(set(class_list) - set(sub_val_list))    
        else:
            print('INFO: using COCO (PANet)')
            class_list = list(range(1, 81))
            if args.split == 3:
                sub_list = list(range(1, 61))
                sub_val_list = list(range(61, 81))
            elif args.split == 2:
                sub_list = list(range(1, 41)) + list(range(61, 81))
                sub_val_list = list(range(41, 61))
            elif args.split == 1:
                sub_list = list(range(1, 21)) + list(range(41, 81))
                sub_val_list = list(range(21, 41))
            elif args.split == 0:
                sub_list = list(range(21, 81)) 
                sub_val_list = list(range(1, 21))
                
    return sub_list, sub_val_list

def is_same_model(model1, model2):
    flag = 0
    count = 0
    for k, v in model1.state_dict().items():
        model1_val = v
        model2_val = model2.state_dict()[k]
        if (model1_val==model2_val).all():
            pass
        else:
            flag+=1
            print('value of key <{}> mismatch'.format(k))
        count+=1

    return True if flag==0 else False

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def freeze_modules(model):
    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():  # 1024-d
        param.requires_grad = False
    # for param in model.classifier.parameters():  # 1024-d
    #     param.requires_grad = False
    if hasattr(model,'layer4'):
        print('layer4 of backbone is adopted...')
        for param in model.layer4.parameters():
            param.requires_grad = False

def sum_list(list):
    sum = 0
    for item in list:
        sum += item
    return sum


# ------------------------- Plot -------------------------

# 双变量<蜂巢图>
def plot_distribution(file_path='size_iou.txt'):

    # plt.style.use(['science']) # science notebook scatter
    sns.set_theme(style='darkgrid')
    sns.set_style('ticks')
    plt.rc('font',family='Times New Roman') 

    list_read = open(file_path).readlines()
    miou_list, size_list = [], []
    # for item in list_read[1000*(id-1):1000*id]:
    for item in list_read[6000:7000]:
        miou_temp = float(item.split('\t')[-2])
        size_temp = int(item.split('\t')[-1][:-1])
        # if miou_temp>0.02:
        # if (miou_temp>0.0) & (size_temp<120000):
            # miou_list.append(miou_temp)
            # size_list.append(size_temp)
        miou_list.append(miou_temp)
        size_list.append(size_temp)
    miou_array = np.array(miou_list) 
    size_array = np.array(size_list) 
    data_array = np.concatenate((miou_array.reshape(-1,1),size_array.reshape(-1,1)),1)
    # sns.set(font_scale=1.5)
    # sns.palplot(sns.color_palette("Set2", 10))

    df = pd.DataFrame(data_array, columns=['mIoU', 'Object Size'])
    sns_plot = sns.jointplot(x='Object Size', y='mIoU', data=df, \
                            kind='hex', color='#4CB391', \
                            xlim=(-0.3e4,18.3e4), ylim=(-0.03,1.03), \
                            height=6, ratio=5, space=0.1, \
                            marginal_kws = dict(bins=25, kde=True), \
                            joint_kws = dict(gridsize=25)                            
                            )
    # #4CB391
    # 科学计数法
    def formatnum(x, pos):
        if x > 1e5:
            return '$%.1f$x$10^{5}$' % (x/1e5)
        if x == 1e5:
            return '$%d$x$10^{5}$' % (x/1e5)            
        elif x == 0:
            return '$%d$' % (x)
        else:
            return '$%d$x$10^{4}$' % (x/1e4)            
    formatter = FuncFormatter(formatnum)
    sns_plot.ax_joint.xaxis.set_major_formatter(formatter)

    # 坐标轴字体
    sns_plot.ax_joint.set_ylabel('mIoU', family='Times New Roman', \
                                fontsize=14, fontweight='bold')    
    sns_plot.ax_joint.set_xlabel('Object Size', family='Times New Roman', \
                                fontsize=14, fontweight='bold')

    # 刻度字体
    # labels = sns_plot.ax_joint.get_xticklabels() + sns_plot.ax_joint.get_yticklabels()
    # [label.set_fontname('Times New Roman') for label in labels]
    
    # 标题
    sns_plot.ax_marg_x.set_title('Baseline', fontsize=18, fontweight='bold', family='Times New Roman', color='k') # Baseline 

    # sns.despine()
    # sns_plot.savefig('Distribution.png')
    # sns_plot.savefig('Distribution{}_{}.png'.format(id1,id2), dpi=600)
    sns_plot.savefig('Distribution2_3.png', dpi=600)


    print('done!')
    color_list_r = [252, 60, 60]
    color_list_b = [0,0,150]
    img_tmp = image.copy()
    for c in range(3):
        img_tmp[:, :, c] = np.where(label_p[:,:] == 255,
                                    image[:, :, c] * 0.5 + 0.5 * color_list_r[c],
                                    img_tmp[:, :, c]*0.9)
    for c in range(3):
        img_tmp[:, :, c] = np.where(label_b[:,:] == 255,
                                    image[:, :, c] * 0.5 + 0.5 * color_list_b[c],
                                    img_tmp[:, :, c]*0.9)
    # img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_RGB2BGR)
    img_tmp = cv2.resize(img_tmp, dsize=(500, 500), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('000.png', img_tmp)    

def plot_seg_result(img, mask, type=None, size=500, alpha=0.5, anns='mask'):
    assert type in ['pre', 'gt', 'sup']
    if type == 'pre' or type == 'gt':
        color = (255, 50, 50)     # red  (255, 50, 50) (255, 90, 90) (252, 60, 60)
    elif type == 'sup':
        color = (90, 90, 218)   # blue (102, 140, 255) (90, 90, 218) (90, 154, 218)
    # elif type == 'gt':
    #     color = (255, 218, 90)  # yellow
    color_scribble = (255, 218, 90) # (255, 218, 90) (0, 0, 255)

    img_pre = img.copy()

    if anns == 'mask':
        for c in range(3):
            img_pre[:, :, c] = np.where(mask[:,:,0] == 1,
                                        img[:, :, c] * (1 - alpha) + alpha * color[c],
                                        img[:, :, c])            
    elif anns == 'scribble':
        mask[mask==255]=0
        mask = mask[:,:,0]
        dilated_size = 5
        Scribble_Expert = ScribblesRobot()
        scribble_mask = Scribble_Expert.generate_scribbles(mask)
        scribble_mask = ndimage.maximum_filter(scribble_mask, size=dilated_size) # 
        for c in range(3):
            img_pre[:, :, c] = np.where(scribble_mask == 1,
                                        color_scribble[c],
                                        img[:, :, c])                    
    elif anns == 'bbox':
        mask[mask==255]=0
        mask = mask[:,:,0]        
        bboxs = find_bbox(mask)
        for j in bboxs: 
            cv2.rectangle(img_pre, (j[0], j[1]), (j[0] + j[2], j[1] + j[3]), (255, 0, 0), 4) # -1->fill; 2->draw_rec

    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_RGB2BGR)  
    
    if size is not None:
        img_pre = cv2.resize(img_pre, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    return img_pre
    # cv2.imwrite('{}.bmp'.format(type), img_pre)

def plot_seg_result_Nway(img, gt_mask, pre_mask, size=500, alpha=0.5):

    idx_list_gt = list(np.unique(gt_mask))
    assert len(idx_list_gt) >= 3
    
    color_list = [(255, 50, 50), (102, 140, 255)] # red & blue
    
    img_gt = img.copy()
    img_pre = img.copy()

    for idx in idx_list_gt:
        if idx == 255 or idx == 0:
            continue
        for c in range(3):
            img_gt[:, :, c] = np.where(gt_mask[:,:,0] == idx,
                                        img[:, :, c] * (1 - alpha) + alpha * color_list[idx_list_gt.index(idx)-1][c],
                                        img_gt[:, :, c])
            img_pre[:, :, c] = np.where(pre_mask[:,:,0] == idx,
                                        img[:, :, c] * (1 - alpha) + alpha * color_list[idx_list_gt.index(idx)-1][c],
                                        img_pre[:, :, c])

    img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
    img_pre = cv2.cvtColor(img_pre, cv2.COLOR_RGB2BGR)
    if size is not None:
        img_gt = cv2.resize(img_gt, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        img_pre = cv2.resize(img_pre, dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    return img_gt, img_pre

def plot_act_map(img, act_map, size=500, alpha=0.5): 

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    # act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min())
    act_map = np.uint8(255*act_map)
    heatmap = cv2.applyColorMap(act_map, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
    zeromap = cv2.applyColorMap(np.zeros_like(act_map), cv2.COLORMAP_JET)
    onemap = cv2.applyColorMap(np.ones_like(act_map)*255, cv2.COLORMAP_JET)
    heat_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)     # 将伪彩色图与原始图片融合
    zero_img = cv2.addWeighted(img, 1-alpha, zeromap, alpha, 0)
    one_img = cv2.addWeighted(img, 1-alpha, onemap, alpha, 0)
    
    if size is not None:
        heat_img = cv2.resize(heat_img, dsize=(size, size), interpolation=cv2.INTER_LINEAR)    
        zero_img = cv2.resize(zero_img, dsize=(size, size), interpolation=cv2.INTER_LINEAR)    
        one_img = cv2.resize(one_img, dsize=(size, size), interpolation=cv2.INTER_LINEAR)    
    # cv2.imwrite('555.bmp', heat_img)
    return heat_img, zero_img, one_img

def plot_cat_wise_result():
    cat_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car' , 'cat', 'chair', 'cow',
                'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
                'train', 'tv/monitor']
    idx = ['Baseline', 'Ours']
    # Ori Data
    # baseline_list = [0.7893,0.3473,0.7330,0.5328,0.4497,0.8482,0.5905,0.8452,0.2091,0.8547,0.2337,0.8106,0.8079,0.7451,0.5194,0.2688,0.8667,0.3990,0.7119,0.3431]
    # our_list =      [0.7977,0.3538,0.7237,0.6074,0.5170,0.8591,0.6135,0.8384,0.2772,0.8761,0.2760,0.8104,0.8071,0.7652,0.5175,0.2752,0.8765,0.4247,0.7538,0.3387]

    baseline_list = [0.7893,0.3473,0.7237,0.5328,0.4497,0.8482,0.5905,0.8384,0.2091,0.8547,0.2337,0.8106,0.8071,0.7451,0.5175,0.2688,0.8667,0.3990,0.7119,0.3387]
    # our_list =      [0.7977,0.3538,0.7330,0.6074,0.5170,0.8591,0.6135,0.8452,0.2772,0.8761,0.2760,0.8104,0.8079,0.7652,0.5194,0.2752,0.8765,0.4247,0.7538,0.3431]
    margin_list = [0.0144, 0.0385, 0.0163, 0.0546, 0.0673, 0.0159, 0.038, 0.0118, 0.0681, 0.0214, 0.0373, 0.0108, 0.0158, 0.0171, 0.0299, 0.0214, 0.0098, 0.0307, 0.0419, 0.0094]
    data = np.concatenate((np.array(baseline_list).reshape(1,-1),np.array(margin_list).reshape(1,-1)),axis=0)
    df = pd.DataFrame(data,index=idx,columns=cat_names)

    x = np.arange(20)
    # plt.style.use(['science'])
    plt.rc('font',family='Times New Roman') 
    fig, ax = plt.subplots()
    # 取消边框
    # for key, spine in ax.spines.items():
    #     # 'left', 'right', 'bottom', 'top'
    #     if key == 'right' or key == 'top':
    #         spine.set_visible(False)
    plt.bar(x, data[0], color='steelblue', edgecolor='k', lw=0.5, tick_label=cat_names, label="Baseline") # #66c2a5
    plt.bar(x, data[1], bottom=data[0], color='#A4BAD9', edgecolor='k', lw=0.5, label="Ours")           # #8da0cb

    # plt.bar(x, data[1], tick_label=cat_names, color='#A4BAD9')           # Single

    plt.legend(idx, loc=4)
    plt.xticks(rotation=45)
    plt.tick_params(labelsize=9)
    plt.xlim(-1,20)
    plt.ylim(0.18,0.96)
    plt.ylabel('mIoU', fontsize=12, fontweight='bold')
    plt.grid(linestyle='--', linewidth=0.2)

    y_shift = 0 # 0.04
    # for xx, yy1, yy2 in zip(x,np.array(baseline_list),np.array(our_list)):
    #     plt.text(xx,yy1-y_shift, '%.4f'%yy1, ha='center', va='bottom', size=5)
    #     plt.text(xx,yy2, '%.4f'%yy2, ha='center', va='bottom', size=5)

    plt.savefig('Cat_wise.png', bbox_inches='tight', dpi=600) # dpi=600
    plt.close()

    print('done!')

def plot_loss():

    file_path = 'split0_vggloss.txt'
    list_read = open(file_path).readlines()
    iter_list = []
    epoch_list = []
    event_list = []
    loss_list = []
    epoch_pre = 1
    num = 0
    main, aux1, aux2 = 0, 0, 0
    for id, item in tqdm(enumerate(list_read)):
        iter = id+1
        epoch = int(item.split(' ')[0])
        if epoch == epoch_pre:
            main += float(item.split(' ')[1])
            aux1 += float(item.split(' ')[2])
            aux2 += float(item.split(' ')[3][:-1])    
            num += 1        
            continue
        else:
            main /= num
            aux1 /= num
            aux2 /= num      
            for i in range(3):
                iter_list.append(iter)
                epoch_list.append(epoch_pre)
                if i == 0:
                    loss_list.append(main)
                    event_list.append('main')
                elif i == 1:
                    loss_list.append(aux1)
                    event_list.append('aux1')
                elif i == 2:
                    loss_list.append(aux2)
                    event_list.append('aux2')

            epoch_pre = epoch
            num, main, axu1, aux2 = 1, 0, 0, 0
            main += float(item.split(' ')[1])
            aux1 += float(item.split(' ')[2])
            aux2 += float(item.split(' ')[3][:-1])

    sns.set_theme(style="darkgrid", font='Palatino Linotype', font_scale=1.2)
    # sns.set_style({'axes.spines.left': False})
    pd_data = {'Epochs':epoch_list, 'Loss':loss_list, 'loss':event_list}
    df = pd.DataFrame(pd_data)
    plt.figure(figsize=(8, 4))
    sns_plot = sns.lineplot(x='Epochs',y='Loss', hue='loss', data=df, linewidth=3, legend=None)
    # plt.legend(labels = ['main','aux1', 'aux2'], loc = 1, bbox_to_anchor = (1,1))
    plt.xlim(-5,140)
    fig = sns_plot.get_figure()
    fig.savefig('loss.png', bbox_inches ='tight', dpi=600)
    plt.close()

    print('done!')

if __name__ == '__main__':
    plot_loss()
    # file_path = 'size_iou.txt'
    # for id in range(1,9):
    #     plot_distribution(id, file_path)
    # plot_distribution(file_path)

    # plot_cat_wise_result()