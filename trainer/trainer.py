import torch


class ModelTrainer:
    """
    Args:
        model : An end to end trainable model.
        optimizer (nn.optim.Optimizer): optimizer used for training

        loss_criterion (callable) : loss function
        train_loader/val_loader (torch.utils.data.DataLoader) : training data loader
    """

    def __init__(self, model, optimizer, lr_scheduler, train_loader, loss_criterion, parameter_dir, val_loader=None,
                 val_criterion=None, epochs=300, num_iterations=1, validate_after_iter=100):
        self.model = model.cuda()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.loss_criterion = loss_criterion
        self.parameter_dir = parameter_dir
        self.val_loader = val_loader
        self.val_criterion = val_criterion
        self.epochs = epochs
        self.num_iterations = num_iterations
        self.validate_after_iter = validate_after_iter

    def start_train(self):
        epochs = self.epochs
        for i in range(epochs):
            self.train()
 
    def train(self):
        """
        训练一个epoch，每次训练均进行验证，验证结果保存在名称中。

        Args:
        Return:
        """

        # set model in training model
        self.model.train()
        train_loss = RunningAverage()
        for i, data in enumerate(self.train_loader):
            input_data, label = data['image'].float().cuda(), data['label'].cuda()
            # forward pass
            output = self.model(input_data)
            # print(output.size())
            # compute the loss
            loss = self.loss_criterion(output, label)
            train_loss.update(loss.item())
            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.val_loader is not None:
                if self.num_iterations % self.validate_after_iter == 0:
                    # set the model in eval mode
                    self.model.eval()
                    # evaluate on validation set
                    val_score = self.validate(self.val_loader)
                    print(val_score)
                    # set the model back to training mode
                    self.model.train()
                    if val_score >= 0.8:
                        # 当系数大于80%时保存模型参数
                        place = self.parameter_dir + str(val_score)
                        torch.save(self.model.state_dict(), place)

            self.num_iterations += 1
        print('epoch finished, epoch_loss:{}'.format(train_loss.avg))

    def validate(self, val_loader):
        val_scores = RunningAverage()
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                eval_data, eval_label = val_data['image'].float().cuda(), val_data['label'].cuda()
                output = self.model(eval_data)
                # 模型最后一层需要进行softmax
                eval_score = self.val_criterion(output, eval_label)
                val_scores.update(eval_score.item())
        return val_scores.avg


class RunningAverage:
    """
    Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count