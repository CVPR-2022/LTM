# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from .mcmf import *


def replace_base_fc(trainset, transform, model, args):    
    # replace fc.weight with the embedding average of train data        
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,    
                                              num_workers=8, pin_memory=True, shuffle=False)    
    trainloader.dataset.transform = transform    
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding,_ = model(data) 
            embedding_list.append(embedding.cpu()) 
            label_list.append(label.cpu()) 
    embedding_list = torch.cat(embedding_list, dim=0) 
    label_list = torch.cat(label_list, dim=0) 
    
    proto_list = []
    cov_list = []
    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        #
        cov_this = torch.tensor(np.cov(embedding_this.cpu().numpy().T))
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)
        cov_list.append(cov_this)
    proto_list = torch.stack(proto_list, dim=0) 
    cov_list = torch.stack(cov_list, dim=0) 
    print('cov_list', cov_list.shape) 

    model.module.fc.weight.data[:args.base_class] = proto_list 
    model.module.mem_feat[:args.base_class,:] = proto_list  
    model.module.mem_cov[:args.base_class] = cov_list  
    model = model . train()  
    
    return model  
    
def generate_prototypes(args, model):
    print('start generating samples for base classes', len(model.module.g_samples))
    if args.dataset in ['cifar100', 'mini_imagenet']:
        count = 60
    else:
        count=100
    for idx in tqdm(range(count)):
        sam_tmp = np.random.multivariate_normal(mean=model.module.mem_feat[idx,:].cpu().numpy(), cov=model.module.mem_cov[idx,:].cpu().numpy(), size=10)
        model.module.g_samples.append(sam_tmp)
    print('fininshed generating samples for base classes', len(model.module.g_samples))
        
    
def dis_train(args,model):
    dis_loss = 0
#     print('length of dis list', len(model.module.dis_list))
    for idx,sub_list in enumerate(model.module.dis_list):
#         sam_tmp = torch.tensor(model.module.mem_feat[idx,:], dtype=torch.float32).cuda().unsqueeze(0)

# gussian sampling from prototypes
#         sam_tmp = np.random.multivariate_normal(mean=model.module.mem_feat[idx,:].cpu().numpy(), cov=model.module.mem_cov[idx,:].cpu().numpy(), size=10)
        sam_tmp = model.module.g_samples[idx]
        sam_tmp = torch.Tensor(sam_tmp).cuda()
        
        sam_tmp = model.module.layer1(sam_tmp)
        
        sam_tmp = model.module.layer2(sam_tmp)
#         sam_tmp = model.module.layer3(sam_tmp)
#         sam_tmp = model.module.layer4(sam_tmp)
#         sam_tmp = model.module.layer4(sam_tmp)
        sam_cat = sam_tmp
        # get logits
        sam_tmp = F.linear(F.normalize(sam_tmp, p=2, dim=-1), F.normalize(model.module.fc.weight, p=2, dim=-1))
        sam_tmp = model.module.args.temperature * sam_tmp
        sam_logits = sam_tmp[:, :args.base_class]
        sam_label = idx*torch.ones(sam_tmp.shape[0]).long().cuda()
#         print('shape of sam_label', sam_label.shape)
        sam_loss = F.cross_entropy(sam_logits, sam_label)
        
        # matching loss
        
        
#         print('shape of sam_tmp', sam_cat.shape)
#         sam_cat = sam_tmp[:,:512]
#         for j in range(49):
#             sam_cat = torch.cat([sam_cat, sam_tmp[:,512*(j+1):512*(j+2)]],dim=0)
#         print('length of sub_list', len(sub_list))   
        
#         if len(sub_list)>2:  
        sub_tensor = torch.tensor(sub_list, dtype=torch.float32).cuda()          
#         print('check shape', sam_cat.shape, sub_tensor.shape)     
        sam_cat, sub_tensor = MCMFMatch(sam_cat, sub_tensor) 
#         print('shape of sam_cat and sub_tensor', sam_cat.shape, sub_tensor.shape) 
        dis_loss += torch.cosine_similarity(sam_cat, sub_tensor, dim=1).sum(1).sum(0) 
        
        inter_loss = torch.cosine_similarity(sub_tensor[0,:5,:], sub_tensor[0,5:,:],dim=0).sum(0)    
#         print('shape of inter_loss', inter_loss.shape)      
#         print('shape of sam_cat', sam_cat.shape) 
#         for k in range(sam_cat.shape[1]): 
#             for n in range(k): 
#                 if k!=n: 
#                     tmp_loss = torch.cosine_similarity(sub_tensor[0,k,:], sub_tensor[0,n,:],dim=0) 
#                     inter_loss += tmp_loss  
        dis_loss = dis_loss - inter_loss  
                    
#             dis_loss = dis_loss.sum(1).sum(0)  
        
#         for item in sub_list:  
# #             sam_tmp = torch.tensor(np.random.multivariate_normal(mean=model.module.mem_feat[idx,:].cpu().numpy(), cov=model.module.mem_cov[idx,:,:].cpu().numpy(), size=1))  
#             sam_tmp = torch.tensor(model.module.mem_feat[idx,:], dtype=torch.float32).cuda()  
#             sam_tmp = model.module.layer1(sam_tmp)     
#             sam_tmp = model.module.layer2(sam_tmp)     
# #             item = torch.tensor(item, dtype=torch.float32) 
# #             sam_tmp = torch.tensor(sam_tmp, dtype=torch.float32)  
# #             print('shape of genereted samples', item.shape, sam_tmp.shape)  
#             sam_tmp, item = MCMFMatch(sam_tmp, item.unsqueeze(0))  
#             print('match finished') 
#             dis_loss += torch.cosine_similarity(item, sam_tmp.squeeze(0), dim=0)  
        
    return dis_loss, sam_loss


def base_train(model, trainloader, optimizer, scheduler, epoch, args, dis_flag = False, g_label_flag = False):
    if args.dataset in ['cifar100','mini_imagenet']:
        count = 60
    else:
        count=100
    total_loss = 0
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain

    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        
        logits, g_label = model(data)
        if g_label_flag and i==1:
            for i in range(count):
                model.module.dis_list.append(list())
        for i in range(train_label.shape[0]):
            if len(model.module.dis_list[train_label[i]])<10:
                model.module.dis_list[train_label[i]].append(g_label[i,:].tolist())
#         print('length of dis_list', len(model.module.dis_list))
        logits = logits[:, :args.base_class]
        
#         print('check train label', train_label)
#         model.module.
########
#         for i in range(60):
            
#         mean = mem_feat[i,:].cpu().numpy()    #network.mem_feat[i,:].cpu().numpy()
#         cov = mem_cov[i,:,:].cpu().numpy()    #network.mem_cov[i,:,:].cpu().numpy()
#         if i<60:
#             sampled_feat = np.random.multivariate_normal(mean=mean, cov=cov, size=10) #*0.0
            
########        
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        
        if dis_flag:
            dis_loss, sam_loss = dis_train(args, model)
#             print('dis loss', dis_loss)
            loss =  loss + sam_loss + 0.5*dis_loss
#             print('sam loss', sam_loss)
            
        else:
            loss = loss #+ sam_loss
            
        total_loss = loss
        
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    return tl, ta

        

def test(model, testloader, epoch, args, session):
    ##
    gs_list = list()
    for idx in range(90+session*10, 100+session*10):
        sam_tmp = np.random.multivariate_normal(mean=model.module.mem_feat[idx,:].cpu().numpy(), cov=model.module.mem_cov[idx,:].cpu().numpy(), size=30)
        sam_tmp = torch.Tensor(sam_tmp).cuda()
#         sam_tmp = model.module.layer1(sam_tmp)
#         sam_tmp = model.module.layer2(sam_tmp)
        gs_list.append(sam_tmp)
        
    ##
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    acc_list = list()
    tsne_list = list()
    label_list = list()
    cls_list = list()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, test_label = [_.cuda() for _ in batch]
#             print('test data shape:', data.shape, test_label.shape)
            logits, tsne_feat = model(data)   
            tsne_list.append(tsne_feat)
            label_list.append(test_label)
            logits = logits[:, :test_class] 
            cls_list.append(torch.argmax(logits, dim=1))
#             print('shape of logits', torch.argmax(logits, dim=1))
            loss = F.cross_entropy(logits, test_label)  
            acc = count_acc(logits, test_label)  

            vl.add(loss.item())
            va.add(acc)
            acc_list.append(acc)
#             print('accuracy of class %d :' % test_label.unique(), acc)

        vl = vl.item()
        va = va.item()
    print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    return vl, va, acc_list, tsne_list, label_list, gs_list, cls_list



