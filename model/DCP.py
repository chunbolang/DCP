import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP, ASPP_Drop ,ASPP_BN



def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005  
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()


        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.pretrained = True
        self.classes = 2

        self.data_set = args.data_set
        self.split = args.split
        
        self.map_mode = 'Cosine'
        assert self.layers in [50, 101, 152]

        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('INFO: Using ResNet {}'.format(self.layers))
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, 1, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + 4, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        
        #parallel decoder structure
        self.ASPP = ASPP()

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )      

        self.init_merge_bg = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + 3 , reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

        self.ASPP_bg = ASPP()
        self.res1_bg = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2_bg = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )      
        self.cls_bg = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, 1, kernel_size=1)
        )             

        self.GAP = nn.AdaptiveAvgPool2d(1)
        
        # self-reasoning scheme
        if self.vgg:
            self.res1_simple = nn.Sequential(
            nn.Conv2d(512 , reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
            )    

            self.res2_simple = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
                )
            self.cls_simple = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            )
             
        else:
            self.res1_simple = nn.Sequential(
            nn.Conv2d(2048 , reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
            )    

            self.res2_simple = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
                )
            self.cls_simple = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            )



    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
            {'params': model.down_query.parameters()},
            {'params': model.down_supp.parameters()},
            {'params': model.init_merge.parameters()},
            {'params': model.ASPP.parameters()},
            {'params': model.res1.parameters()},
            {'params': model.res2.parameters()},      
            {'params': model.cls_simple.parameters()},   
            {'params': model.cls.parameters()},
            {'params': model.cls_bg.parameters()},
            {'params': model.init_merge_bg.parameters()},
            {'params': model.ASPP_bg.parameters()},
            {'params': model.res1_bg.parameters()},
            {'params': model.res2_bg.parameters()},
            {'params': model.res1_simple.parameters()},
            {'params': model.res2_simple.parameters()}
            ],

            lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)  # 2.5e-3, 0.9, 1e-4
        
        return optimizer


    def forward(self, x, s_x, s_y, y, cat_idx=None):
    
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)    # 473

#   Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)                # [4, 128, 119, 119]
            query_feat_1 = self.layer1(query_feat_0)     # [4, 256, 119, 119]
            query_feat_2 = self.layer2(query_feat_1)     # [4, 512, 60, 60]
            query_feat_3 = self.layer3(query_feat_2)     # [4, 1024, 60, 60]
            query_feat_4 = self.layer4(query_feat_3)     # [4, 2048, 60, 60]
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)  # [4, 1536, 60, 60]
        query_feat = self.down_query(query_feat)                 # [4, 256, 60, 60]

#   Support Feature     
        supp_feat_list = []
        supp_feat_simple_list = []
        final_supp_list = []
        supp_simple_out_list = []
        mask_list = []
        supp_feat_alpha_list = []
        supp_feat_gamma_list = []
        supp_feat_delta_list = []
        supp_feat_beta_list = []
        supp_feat_BG_list = []

        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)   # [4, 1024, 60, 60]
                supp_feat_4_true = self.layer4(supp_feat_3)
                

                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                mask_1 = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
                supp_feat_4 = self.layer4(supp_feat_3*mask)  
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)

            supp_simple_out = self.res1_simple(supp_feat_4_true)
            supp_simple_out = self.res2_simple(supp_simple_out) + supp_simple_out
            supp_simple_out = self.cls_simple(supp_simple_out)

            supp_simple_out_list.append(supp_simple_out)  #simple
            mask_simple = F.interpolate(supp_simple_out, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            mask_simple_pre = mask_simple.max(1)[1].unsqueeze(1)  # 60*60
            mask_alpha = mask_simple_pre * mask_1
            mask_delta = mask_simple_pre - mask_alpha
            mask_beta = mask_1 - mask_alpha
            mask_gamma = 1- mask_alpha - mask_delta - mask_beta
            mask_BG = 1 - mask_1


            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat_tem = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat_tem, mask)
            supp_feat_alpha = Weighted_GAP(supp_feat_tem, mask_alpha)
            supp_feat_delta = Weighted_GAP(supp_feat_tem, mask_delta)
            supp_feat_gamma = Weighted_GAP(supp_feat_tem, mask_gamma)
            supp_feat_beta = Weighted_GAP(supp_feat_tem, mask_beta)
            supp_feat_BG = Weighted_GAP(supp_feat_tem, mask_BG)

            supp_feat_alpha_list.append(supp_feat_alpha)
            supp_feat_gamma_list.append(supp_feat_gamma)
            supp_feat_delta_list.append(supp_feat_delta)
            supp_feat_beta_list.append(supp_feat_beta)
            supp_feat_BG_list.append(supp_feat_BG)

            supp_feat_simple = Weighted_GAP(supp_feat_tem, mask_simple[:,1,:,:].unsqueeze(1))
            supp_feat_list.append(supp_feat)    # [4, 256, 1, 1] 
            supp_feat_simple_list.append(supp_feat_simple)

# corrquery
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list): 
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]  

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1) 
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)    
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)  #
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  

# k_shot
        if self.shot > 1:             
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list) 

            supp_feat_simple = supp_feat_simple_list[0]
            for i in range(1, len(supp_feat_simple_list)):
                supp_feat_simple += supp_feat_simple_list[i]
            supp_feat_simple /= len(supp_feat_simple_list) 

            supp_feat_alpha = supp_feat_alpha_list[0]
            for i in range(1, self.shot):
                supp_feat_alpha += supp_feat_alpha_list[i]
            supp_feat_alpha /= self.shot

            supp_feat_gamma = supp_feat_gamma_list[0]
            for i in range(1, self.shot):
                supp_feat_gamma += supp_feat_gamma_list[i]
            supp_feat_gamma /=self.shot

            supp_feat_beta = supp_feat_beta_list[0]
            for i in range(1, self.shot):
                supp_feat_beta += supp_feat_beta_list[i]
            supp_feat_beta /=self.shot

            supp_feat_delta = supp_feat_delta_list[0]
            for i in range(1, self.shot):
                supp_feat_delta += supp_feat_delta_list[i]
            supp_feat_delta /=self.shot

            supp_feat_BG = supp_feat_BG_list[0]
            for i in range(1, self.shot):
                supp_feat_BG += supp_feat_BG_list[i]
            supp_feat_BG /=self.shot

        pro_alpha = supp_feat_alpha
        pro_beta = supp_feat_beta
        pro_gamma = supp_feat_gamma
        pro_delta = supp_feat_delta
        pro_BG = supp_feat_BG

        
        pro_map = torch.cat([pro_BG.unsqueeze(1) , pro_alpha.unsqueeze(1) , \
                supp_feat.unsqueeze(1) , pro_beta.unsqueeze(1) , pro_delta.unsqueeze(1), pro_gamma.unsqueeze(1)], 1)
        activation_map = self.query_region_activate(query_feat, pro_map , self.map_mode).unsqueeze(2) #b,5,1,h,w
        # 0-BG ,1-alpha, 2-supp_feat, 3-beta, 4-delta , 5_gamma

# self-reasoning scheme

        query_simple_out = self.res1_simple(query_feat_4)
        query_simple_out = self.res2_simple(query_simple_out) + query_simple_out
        query_simple_out = self.cls_simple(query_simple_out)
        
        feat_fuse = supp_feat

        query_feat_bin = query_feat
        supp_feat_bin = feat_fuse.expand(-1, -1, query_feat.size(2), query_feat.size(3))
        supp_feat_bin_BG = pro_BG.expand(-1, -1, query_feat.size(2), query_feat.size(3))

        corr_mask_bin = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  
        
        merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin, activation_map[:,1,...],activation_map[:,2,...], activation_map[:,3,...]], 1)   # 256+256+1+1
        merge_feat_bin = self.init_merge(merge_feat_bin)     

        merge_feat_bin_BG = torch.cat([query_feat_bin, supp_feat_bin_BG, activation_map[:,0,...], activation_map[:,4,...], activation_map[:,5,...]], 1)   # 256+256+1+1
        merge_feat_bin_BG = self.init_merge_bg(merge_feat_bin_BG)  

        query_feat = self.ASPP(merge_feat_bin)
        query_feat = self.res1(query_feat)               
        query_feat = self.res2(query_feat) + query_feat  
        out = self.cls(query_feat)                       

        BG_feat = self.ASPP_bg(merge_feat_bin_BG)
        BG_feat = self.res1_bg(BG_feat)
        BG_feat = self.res2_bg(BG_feat) + BG_feat
        out_BG = self.cls_bg(BG_feat)

        output_fin = torch.cat([out_BG, out], 1)
        

        #   Oualphaut Part

        if self.zoom_factor != 1:
            output_fin = F.interpolate(output_fin, size=(h, w), mode='bilinear', align_corners=True)
            query_simple_out = F.interpolate(query_simple_out, size=(h, w), mode='bilinear', align_corners=True)
            for i in range(self.shot):
                supp_simple_out_list[i]  = F.interpolate(supp_simple_out_list[i] , size=(h, w), mode='bilinear', align_corners=True)
                mask_list[i] = F.interpolate(mask_list[i], size=(h, w), mode='bilinear', align_corners=True)
            

        if self.training:
            main_loss = self.criterion(output_fin, y.long())
            aux_loss = self.criterion(query_simple_out, y.long())  
            for i in range(self.shot):
                aux_loss += self.criterion(supp_simple_out_list[i], mask_list[i].squeeze(1).long())
            aux_loss = aux_loss/(1+self.shot)
            return output_fin.max(1)[1], main_loss, aux_loss
        else:
            return output_fin

    def query_region_activate(self, query_fea, prototypes, mode):
        """             
        Input:  query_fea:      [b, c, h, w]
                prototypes:     [b, n, c, 1, 1]
                mode:           Cosine/Conv/Learnable
        Oualphaut: activation_map: [b, n, h, w]
        """
        b, c, h, w = query_fea.shape
        n = prototypes.shape[1]
        que_temp = query_fea.reshape(b, c, h*w)

        if mode == 'Conv':
            map_temp = torch.bmm(prototypes.squeeze(-1).squeeze(-1), que_temp)  # [b, n, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            return activation_map

        elif mode == 'Cosine':
            que_temp = que_temp.unsqueeze(dim=1)           # [b, 1, c, h*w]
            prototypes_temp = prototypes.squeeze(dim=-1)   # [b, n, c, 1]
            map_temp = nn.CosineSimilarity(2)(que_temp, prototypes_temp)  # [n, c, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            activation_map = (activation_map+1)/2          # Normalize to (0,1)
            return activation_map

        elif mode == 'Learnable':
            for p_id in range(n):
                prototypes_temp = prototypes[:,p_id,:,:,:]                         # [b, c, 1, 1]
                prototypes_temp = prototypes_temp.expand(b, c, h, w)
                concat_fea = torch.cat([query_fea, prototypes_temp], dim=1)        # [b, 2c, h, w]                
                if p_id == 0:
                    activation_map = self.relation_coding(concat_fea)              # [b, 1, h, w]
                else:
                    activation_map_temp = self.relation_coding(concat_fea)              # [b, 1, h, w]
                    activation_map = torch.cat([activation_map,activation_map_temp], dim=1)
            return activation_map


