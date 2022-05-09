import os
import os.path as osp
import random
import datetime
import time
import cv2
import numpy as np
import logging
import argparse
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist0
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast as autocast

from model import DCP

from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_parser():

    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='DCP')
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split1_vgg.yaml', help='config file') # pascal/pascal_split0_resnet50.yaml coco/coco_split0_resnet101.yaml
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_model(args):

    model = eval(args.arch).OneModel(args, cls_type='Base')
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.weight:
        weight_path = osp.join(args.snapshot_path, args.weight)
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer

def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    print(args)

    # assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in [0, 1, 2, 3, 999]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    
    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    logger.info(model)
    val_manual_seed = args.manual_seed # <123> 100 200 300 400 <321>
    val_num = 5 # 5
    setup_seed(val_manual_seed, False)
    seed_array = np.random.randint(0,1000,val_num)  

# ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

     # Val
    if args.evaluate:
        if args.resized_val:
            val_transform = transform.Compose([
                transform.Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        else:
            val_transform = transform.Compose([
                transform.test_Resize(size=args.val_size),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
        if args.data_set == 'pascal' or args.data_set == 'coco':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, data_list=args.val_list, \
                                    transform=val_transform, mode='val', \
                                    data_set=args.data_set, use_split_coco=args.use_split_coco)
        elif args.data_set == 'fss':
            val_data = dataset.SemData_fss(shot=args.shot, data_root=args.data_root, data_list=args.val_list, \
                                    transform=val_transform, mode='val', \
                                    data_set=args.data_set)
        elif args.data_set == 'DAVIS':
            val_data = dataset.SemData_DAVIS(shot=args.shot, data_root=args.data_root, data_list=args.val_list, \
                                    transform=val_transform, mode='val', \
                                    data_set=args.data_set)     
                                         
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=False, sampler=None)


# ----------------------  VAL  ----------------------
    start_time = time.time()
    FBIoU_array = np.zeros(val_num)
    mIoU_array = np.zeros(val_num)
    pIoU_array = np.zeros(val_num)

    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id+1, val_num, val_seed))
        loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou, pIoU = validate(val_loader, model, val_seed) 

        FBIoU_array[val_id], mIoU_array[val_id], pIoU_array[val_id] = mIoU_val, class_miou, pIoU
    
    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
    print('FBIoU: {}'.format(np.round(FBIoU_array, 4)))
    print('pIoU:  {}'.format(np.round(pIoU_array, 4)))
    print('-'*43)
    print('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {}'.format(seed_array[mIoU_array.argmax()], seed_array[FBIoU_array.argmax()], seed_array[pIoU_array.argmax()]))
    print('Best_mIoU: {:.4f} \t Best_FBIoU: {:.4f} \t Best_pIoU: {:.4f}'.format(mIoU_array.max(), FBIoU_array.max(), pIoU_array.max()))
    print('Mean_mIoU: {:.4f} \t Mean_FBIoU: {:.4f} \t Mean_pIoU: {:.4f}'.format(mIoU_array.mean(), FBIoU_array.mean(), pIoU_array.mean()))

    with open('./test_record.txt', 'a') as f:
        f.write('\n' + args.arch + ' '*4 + args.weight + '\n')
        f.write('Seed0: {}\n'.format(val_manual_seed))
        f.write('Seed:  {}\n'.format(seed_array))
        f.write('mIoU:  {}\n'.format(np.round(mIoU_array, 4)))
        f.write('FBIoU: {}\n'.format(np.round(FBIoU_array, 4)))
        f.write('pIoU:  {}\n'.format(np.round(pIoU_array, 4)))
        f.write('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {} \n'.format(seed_array[mIoU_array.argmax()], seed_array[FBIoU_array.argmax()], seed_array[pIoU_array.argmax()]))
        f.write('Best_mIoU: {:.4f} \t Best_FBIoU: {:.4f} \t Best_pIoU: {:.4f} \n'.format(mIoU_array.max(), FBIoU_array.max(), pIoU_array.max()))
        f.write('Mean_mIoU: {:.4f} \t Mean_FBIoU: {:.4f} \t Mean_pIoU: {:.4f} \n'.format(mIoU_array.mean(), FBIoU_array.mean(), pIoU_array.mean()))
        f.write('-'*47 + '\n')
        f.write(str(datetime.datetime.now()) + '\n')


def validate(val_loader, model, val_seed):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    if args.data_set == 'pascal':
        test_num = 1000 # 5000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = 1000 # 20000 
        split_gap = 20
    elif args.data_set == 'fss':
        # test_num = len(val_loader)
        test_num = 1000
        split_gap = 240        
    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap  

    setup_seed(val_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end
    
    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num/(len(val_loader)-args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, target, s_input, s_mask, cat_idx_list, ori_label) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            if isinstance(input, list):    # multi-scale test
                scale_len = len(input)
                ori_label = ori_label.cuda(non_blocking=True)                

                for scale_id in range(scale_len):
                    input_temp = input[scale_id].cuda(non_blocking=True)
                    target_temp = target[scale_id].cuda(non_blocking=True)
                    s_input_temp = s_input[scale_id].cuda(non_blocking=True)
                    s_mask_temp = s_mask[scale_id].cuda(non_blocking=True)                    

                    start_time = time.time()
                
                    with torch.no_grad():
                        output_temp = model(s_x=s_input_temp, s_y=s_mask_temp, x=input_temp, y=target_temp, cat_idx=cat_idx_list)
                    model_time.update(time.time() - start_time)

                    if args.ori_resize:
                        longerside = max(ori_label.size(1), ori_label.size(2))
                        backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda')*255
                        backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                        target_temp = backmask.clone().long()

                    output_temp = F.interpolate(output_temp, size=target_temp.size()[1:], mode='bilinear', align_corners=True) 
                    loss_temp = criterion(output_temp, target_temp)    
                    if scale_id == 0: 
                        output = output_temp/scale_len
                        loss = loss_temp/scale_len
                    else:
                        output += output_temp/scale_len
                        loss += loss_temp/scale_len

                output = output.max(1)[1]
                if args.ori_resize:
                    target = target_temp
                else:
                    target = target[1].cuda(non_blocking=True)

            else:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                s_input = s_input.cuda(non_blocking=True)
                s_mask = s_mask.cuda(non_blocking=True)            
                ori_label = ori_label.cuda(non_blocking=True)

                start_time = time.time()
                output = model(s_x=s_input, s_y=s_mask, x=input, y=target, cat_idx=cat_idx_list)
                model_time.update(time.time() - start_time)

                if args.ori_resize:
                    longerside = max(ori_label.size(1), ori_label.size(2))
                    backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda')*255
                    backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                    target = backmask.clone().long()

                output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True) 
                output = output.float()
                loss = criterion(output, target)    

                n = input.size(0)
                loss = torch.mean(loss)

                output = output.max(1)[1]

            # Metric
            intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes+1, args.ignore_label)
            intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
                
            cat_idx_list = cat_idx_list[0].cpu().numpy()[0]
            class_intersection_meter[cat_idx_list] += intersection[1]
            class_union_meter[cat_idx_list] += union[1] 

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), 1)
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % (test_num/100) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))
            # Record mIoU & object size
            with open('./size_iou.txt', 'a') as f:
                f.write('{}\t{:.4f}\t{}\n'.format(iter_num, intersection[1]/union[1], target_meter.val[1]))

    val_time = time.time()-val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i+1, class_iou_class[i]))            
    

    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, iou_class[1]



if __name__ == '__main__':
    main()
