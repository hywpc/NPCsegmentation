# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:24:51 2019

@author: hyw
"""

from torch.utils.data import DataLoader
from net.Vnet1 import *
from data.data import *
from loss.Dice_loss import *
use_gpu = torch.cuda.is_available()

def normal(x):
    x = x.detach().cpu().numpy()
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = torch.from_numpy(x).cuda()
    return x

def train_net(net, net1 = None, epoches = 1000, lr = 1e-4, 
              save_params = True, gpu = use_gpu, class_num = 2):
    print('''
          Start training:
              Epochs:{}
              Learning rate:{}
              Training size{}
              Checkpoings:{}
              CUDA:{}
              '''.format(epoches, lr, len(dataset), str(save_params), str(gpu)))
        
    for epoch in range(epoches):
        print('Starting epoch{}'.format(epoch))
        net.train()
        
        epoch_loss = 0.0
        epoch_loss1 = 0.0
        
        for i, data in enumerate(dataloader):
            input, label = data['image'], data['label']
            
            img_list = [input[:, :, :, :96, :96], 
                        input[:, :, :, :96, 96:],
                        input[:, :, :, 96:, :96],
                        input[:, :, :, 96:, 96:]]
            lbl_list = [label[:, :, :, :96, :96], 
                        label[:, :, :, :96, 96:],
                        label[:, :, :, 96:, :96],
                        label[:, :, :, 96:, 96:]]
            
            if gpu:

                    
            for i in range(len(img_list)):
                input, label = img_list[i], lbl_list[i]

                output = net(input)
                print(outpur.shape)
                output = output.max(1)
                
                output_flat = output.squeeze(0).view(class_num,-1).permute(1, 0)
                label_flat = label.squeeze(0).view(class_num, -1).permute(1, 0)
                
                loss = criterion(output_flat, label_flat)
                epoch_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        

        print('epoch finished, epoch_loss:{}'.format(epoch_loss / 4*i))
        
        if save_params:
            torch.save(net.state_dict(), 'E:\\3.Work\\siat_project\\params_Vnet\\net\\net_params{}.pkl'.format(epoch))
            #torch.save(net1.state_dict(), 'E:\\3.Work\\siat_project\\params_Vnet\\net1\\net_params{}.pkl'.format(epoch))
            #torch.save(net.state_dict(), '/home/hyw/Head_Neck_Seg/parameters/net/net_params{}.pkl'.format(epoch))
            #torch.save(net1.state_dict(), '/home/hyw/Head_Neck_Seg/parameters/net1/net_params{}.pkl'.format(epoch))

if __name__ == '__main__':
    net = Vnet1(start_ch = 32, out_ch = 23)
    #net1 = Vnet1(start_ch = 12, in_ch = 24, out_ch = 23)
    
    #dataset = NPC_dataset('/home/hyw/Head_Neck_Seg/HaN_OAR/train/', Loss = 'Dice')
    dataset = NPC_dataset('E:\\3.Work\\data\\HaN_OAR\\train', Loss = 'Dice')
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle = True,
                            num_workers=0,
                            pin_memory=True,
                            )
    criterion = MulticlassDiceLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 
                            lr = 1e-2, 
                            momentum=0.9,
                            weight_decay = 5e-4
                            )
    train_net(net = net, net1 = None, class_num = 23)