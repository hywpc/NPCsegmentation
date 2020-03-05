import torch.optim as optim
from torch.utils.data import DataLoader
from net.Net_1pool import *
from data.data_anatomy import *
from loss.hybrid_loss import *
from metric.metric import DiceCoefficient
from trainer.trainer import ModelTrainer


if __name__ == '__main__':
    model = AnatomyNet(classes=23)
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
    lr_scheduler = None
    # train_set = NPCDataSet('E:\\3.Work\\data\\HaN_OAR\\train', scatter_label=False, transform=False, crop=True,crop_size=(80, 160, 160), mode='train')
    train_set = NPCDataSet('/home/hyw/data/HaN_OAR/train/', scatter_label=True, transform=False, crop=True, crop_size=(80, 160, 160), mode='train')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    loss_criterion = HybridLoss()
    parameter_dir = 'E:\\3.Work\\siat_project\\params\\Net_1pool'
    # val_set = NPCDataSet('E:\\3.Work\\data\\HaN_OAR\\train', scatter_label=True, transform=False, crop=False,crop_size=(80, 160, 160), mode='test')
    val_set = NPCDataSet('/home/hyw/data/HaN_OAR/validation/', scatter_label=True, transform=False, crop=False, crop_size=(80, 160, 160), mode='test')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    val_criterion = DiceCoefficient()
    epochs = 400
    validate_after_iter = 20
    trainer = ModelTrainer(model, optimizer, lr_scheduler, train_loader, loss_criterion, parameter_dir,
                           val_loader=val_loader, val_criterion=val_criterion,
                           epochs=300, num_iterations=1, validate_after_iter=20)
    trainer.start_train()