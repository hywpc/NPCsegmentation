from torch.utils.data import DataLoader
from net.Net_1pool import *
from data.data_anatomy import *
from loss.hybrid_loss import *

use_gpu = torch.cuda.is_available()


def train_net(net, inputData, epochs=1000, lr=1e-4, save_params=True, gpu=use_gpu):

    print('''
          Start training:
              Epochs: {}
              Learning rate: {}
              Training size: {}
              CUDA: {}
              '''.format(epochs, lr, len(inputData), str(gpu)))
    for epoch in range(epochs):
        print('Starting epoch{}'.format(epoch))
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(inputData):
            input_data, label = data['image'], data['label']

            if gpu:
                net = net.cuda()
                input_data = input_data.float().cuda()
                label = label.cuda()

            output = net(input_data)

            loss = criterion(output, label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch finished, epoch_loss:{}'.format(epoch_loss / i))

        if save_params:
            torch.save(net.state_dict(), 'E:\\3.Work\\siat_project\\params\\Net_1pool\\net_params{}.pkl'.format(epoch))
            # torch.save(net.state_dict(), '/home/hyw/siat_project/params/Net1_1pool/net_params{}.pkl'.format(epoch))


if __name__ == '__main__':
    training_net = AnatomyNet(classes=23)
    epoch = 1000
    learning_rate = 1e-4
    # data_set = NPC_dataset('/home/hyw/data/HaN_OAR/train/', size='medium', label_scatter=True)
    data_set = NPCDataSet('E:\\3.Work\\data\\HaN_OAR\\train', scatter_label=True, transform=True, crop=True, crop_size=(80, 160, 160))
    optimizer = torch.optim.SGD(training_net.parameters(),
                                lr=learning_rate,
                                momentum=0.9,
                                )
    optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
    dataloader = DataLoader(data_set,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True,
                            )
    criterion = HybridLoss()
    train_net(training_net, dataloader, epochs=epoch, lr=learning_rate)