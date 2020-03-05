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

def train_net(net,
              net1,
              epoches = 1000,
              lr = 1e-4,
              save_params = True,
              gpu = use_gpu,
              class_num = 2,
              ):
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
            
            img_list = [data['image1'], data['image2'], data['image3'], data['image4']]
            lbl_list = [data['label1'], data['label2'], data['label3'], data['label4']]
            
            if gpu:
                net = net.cuda()
                net1 = net1.cuda()
                input = input.cuda()
                label = label.cuda()
                for i in range(len(img_list)):
                    img_list[i] = img_list[i].cuda()
                    lbl_list[i] = lbl_list[i].cuda()
                
            output = net(input)
            
            values,  output_max = output.max(1)
            
            output_max = normal(output_max)
            
            if True:
                cat_list = [output_max[:,:, :96, :96], output_max[:,:, 96:, :96], \
                            output_max[:,:, :96, 96:], output_max[:,:, 96:, 96:]]
            
            output = output.squeeze(0)
            output = output.permute(1, 2, 3, 0)
            label = label.squeeze(0)
            label = label.permute(1, 2, 3, 0)
            output_flat = output.view(-1, class_num)
            label_flat = label.view(-1, class_num)
            
            loss = criterion(output_flat, label_flat)
            epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for j in range(len(img_list)):
                input1, label1, input2 = img_list[j], lbl_list[j], cat_list[j]
                input2 = input2.unsqueeze(0).float()
                input1= torch.cat([input1, input2], dim=1)

                output1 = net1(input1).squeeze(0).permute(1, 2, 3, 0)
                label1 = label1.squeeze(0).permute(1, 2, 3, 0)

                output_flat = output1.view(-1, class_num)
                label_flat = label1.view(-1, class_num)
            
                loss = criterion(output_flat, label_flat)
                epoch_loss1 += loss.item()
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                

        print('epoch finished, epoch_loss:{}, epoch_loss1: {}'.format(epoch_loss / i, epoch_loss1/ (4*i)))
        
        if save_params:
            torch.save(net.state_dict(), 'E:\\3.Work\\siat_project\\params_Vnet\\net_params{}.pkl'.format(epoch))
            torch.save(net1.state_dict(), 'E:\\3.Work\\siat_project\\params_Vnet\\net_params{}.pkl'.format(epoch))
            #torch.save(net.state_dict(), '/home/hyw/Head_Neck_Seg/parameters/st16_nodeep/net_params{}.pkl'.format(epoch))
            #torch.save(net1.state_dict(), '/home/hyw/Head_Neck_Seg/parameters/st16_nodeep/net_params{}.pkl'.format(epoch))

if __name__ == '__main__':
    net = Vnet1(start_ch = 12, out_ch = 23)
    net1 = Vnet1(start_ch = 12, in_ch = 2, out_ch = 23)
    
    #dataset = NPC_dataset('/home/hyw/Head_Neck_Seg/HaN_OAR/train/')
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
    train_net(net, net1, class_num = 23)