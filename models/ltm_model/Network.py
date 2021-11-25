import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet18_encoder import *
from models.resnet20_cifar import *
import numpy as np
from tqdm import tqdm
from utils import *

from .helper import *
from utils import *
from dataloader.data_utils import *


class MYNET(nn.Module): 
    def __init__(self, args, mode=None): 
        super().__init__() 
        self.mode = mode 
        self.args = args 
        # self.num_features = 512 
        if self.args.dataset in ['cifar100']:  
            self.encoder = resnet20()  
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']: 
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)   # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        if self.args.dataset in ['cifar100']:
            self.layer1 = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(64, 64, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
        
            
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
            )
        
        
        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)    
        
        self.mem_mode = 'image'    
        if self.args.dataset in ['cifar100']:
            self.features = torch.zeros(200,64).cuda()
            self.covs = torch.zeros(200,64,64).cuda()
        else:
            self.features = torch.zeros(200,512).cuda()
            self.covs = torch.zeros(200,512,512).cuda()
        
        self.gate = nn.Parameter(
                torch.ones(9, device="cuda")*5,
                requires_grad=False)
        self.register_buffer('mem_feat', self.features)
        self.register_buffer('mem_cov', self.covs)
        self.register_buffer('gates', self.gate)
        self.dis_list = list()
        self.g_samples = list()
        
    def forward_metric(self, x):
        x, g_out,_,_ = self.encode(x)
        g_label = x

        ##
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        return x, g_label

    def encode(self, x):
        x, out2, out3 = self.encoder(x)
        g_out = x
        
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        
        out2 = F.adaptive_avg_pool2d(out2, 1)
        out3 = F.adaptive_avg_pool2d(out3, 1)
        return x, g_out, out2, out3

    def forward(self, input):
        if self.mode != 'encoder':
            input, g_label = self.forward_metric(input)
            return input, g_label
    
        elif self.mode == 'encoder':
            input,g_out,_,_ = self.encode(input)
            return input, g_out
        else:
            raise ValueError('Unknown mode')
            
    def update_fc(self,dataloader,class_list,session,memory,data_mem,label_mem,testloader,args):
        i = 0
        for batch in dataloader:
            i+=1
            data, label = [_.cuda() for _ in batch]
            data,_,out2,out3=self.encode(data)
            data = data.detach()
            out2 = out2.detach()
            out3 = out3.detach()
            
        origin_data = data
        
        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            print('update average')
            new_fc = self.update_fc_avg(data, label, class_list)
            self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)
            
        if 'ft' in self.args.new_mode:  # further finetune
            print('start finetuning')
            fc, new_fc, optimizer = self.update_fc_ft(origin_data,new_fc,data,label,session)
            

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        fc = []
        print('check class list!!!!!!!!!', class_list)
        loop_nums = class_list[-1]+1
        print('check loop nums', loop_nums)
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            #
            self.mem_feat[class_index,:]=proto
            tmp_cov = torch.tensor(np.cov(embedding.cpu().numpy().T))
#             print('tmp_cov', tmp_cov.shape)
            self.mem_cov[class_index]=tmp_cov
        #
        for i in class_list:
            self.g_samples.append(np.random.multivariate_normal(mean=self.mem_feat[i,:].cpu().numpy(), cov=self.mem_cov[i,:].cpu().numpy(), size=5))
        print('finished update g_samples', class_list, len(self.g_samples))
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    
    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode: 
            return F.linear(x,fc)        
        elif 'cos' in self.args.new_mode:
#             with torch.no_grad():
            norm_fc = F.normalize(fc, p=2, dim=-1)
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), norm_fc)

    def update_fc_ft(self,origin_data,new_fc,data,label,session):
        print('start finetuning!!!')
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True 
        optimizer = torch.optim.SGD([{'params': new_fc, 'lr':self.args.lr_new},
                                     {'params': self.layer1.parameters(), 'lr':0.1},
                                     {'params': self.layer2.parameters(), 'lr':0.1},
                                    ],  
                                     momentum=0.9, dampening=0.9, weight_decay=0)
#################################################################################################
        with torch.enable_grad():
            g_slist = list()
            for j in range(60+5*session):
                sam_tmp = self.g_samples[j]
                sam_tmp = torch.Tensor(sam_tmp).cuda()
                g_samples = self.layer1(sam_tmp)
                g_samples = self.layer2(g_samples)
                g_slist.append(g_samples)
    
                g_label = j*torch.ones(g_samples.shape[0]).long().cuda()
            
                data = torch.cat([data,g_samples],dim=0)
                label = torch.cat([label,g_label],dim=0)
##############################################################################################
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()       #.detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                for i in range(data.shape[0]//data.shape[0]):
                    logits = self.get_logits(data,fc)  #data[25*i:25*(i+1),:]
                    loss = F.cross_entropy(logits, label)  
                    # add
                    dis_loss = 0 
                    inter_loss = 0
                    for j in range(5):
                        match1, match2 = MCMFMatch(g_slist[55+session*5+j], origin_data[5*j:5*(j+1),:])
                        dis_loss += torch.cosine_similarity(match1, match2, dim=2).sum(1).sum(0)
                    loss = loss + dis_loss # -inter_loss
                    optimizer.zero_grad() 
                    loss.backward(retain_graph=True) 
                    optimizer.step() 
                    pass 
                
        self.fc.weight.data[:self.args.base_class + self.args.way * session, :].copy_(fc.data)

        return fc, new_fc, optimizer 
        

        