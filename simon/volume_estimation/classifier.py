import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable

class Classifier(object):
    """
    Classifier which loads a model, trains and tests it on train and test
    datasets.
    
    Args:
    - model
    - train_loader
    - test_loader
    - cuda
    - dtype
    """
    def __init__(self, model, train_loader, test_loader, cuda, target,
                 is_multi_head=False):
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cuda = cuda
        self.target = target
        
        self.is_multi_head = is_multi_head
        self.agr_train_loss = []
        self.agr_test_loss = []
        self.train_agr_normalized_error = []
        self.test_agr_normalized_error = []
    
    def train(self, num_epoch, lr, grad_clip=None):
        """
        Trains the model 
        
        Input:
            - num_epoch: int nb of epochs
            - lr: float, learning rate
            - grad_clip: float, threshold for the gradient
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr,
                                     betas=(0.9, 0.99), 
                                     weight_decay=1e-2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                              gamma=0.2)
        criterion = nn.L1Loss(reduce=False)
        for epoch in range(num_epoch):
            print('learning rate : {}'.format(scheduler.get_lr()[0]))
            train_loss, nb_steps, normalized_error = 0.0, 0.0, 0.0
            t_epoch_start = time.time()
            for i, data in enumerate(self.train_loader):
                if self.is_multi_head :
                    volume_label = Variable(data[1]['volume'].float())
                    spatial_label = Variable(torch.transpose(torch.stack([data[1]['height'],
                                                                          data[1]['width'], 
                                                                          data[1]['length']]), 
                                                             0, 1).float())
                elif len(self.target) > 1 and (not self.is_multi_head):
                    label = Variable(torch.transpose(torch.stack([data[1]['height'],
                                                                  data[1]['width'], 
                                                                  data[1]['length'],
                                                                  data[1]['volume']]),
                                                     0, 1).float())
                else:
                    label = Variable(data[1][self.target[0]].float())
                dmap = Variable(data[0]['depth_map'].float().unsqueeze(1))
                if self.cuda:
                    dmap = dmap.cuda()
                    if self.is_multi_head:
                        volume_label = volume_label.cuda()
                        spatial_label = spatial_label.cuda()
                    else:
                        label = label.cuda()
                # Zero grad the optimizer
                optimizer.zero_grad()
                if self.is_multi_head:
                    spatial_extents, volume = self.model.forward(dmap)
                else:
                    estimation = self.model.forward(dmap)
                
                # Loss computation
                if self.is_multi_head:
                    spatial_loss = criterion(spatial_extents, spatial_label)
                    volume_loss = criterion(volume, volume_label)
                    loss = spatial_loss.mean() + volume_loss.mean()
                    normalized_error += float((volume_loss / volume_label).mean())
                else:
                    loss = criterion(estimation, label)
                    if self.model.target_size > 1:
                        normalized_error += float((loss[:, -1] / label[:, -1]).mean())
                    else:
                        normalized_error += float((loss / label).mean())
                train_loss += float(loss.mean())
                nb_steps += 1
                
                # Backprop
                loss.mean().backward()
                if grad_clip is not None:
                    for p in self.model.parameters():
                        p.grad.data = torch.clamp(p.grad.data, 
                                                  min=-grad_clip, 
                                                  max=grad_clip).clone()
                optimizer.step()
                t_epoch = time.time()
                if i % 20 == 19:
                    print('epoch: {}, step: {} out of {}, running average train loss'
                          ': {}, running train normalized error: {},'
                          'elapsed time: {}'.format(epoch + 1, i + 1, len(self.train_loader),
                                                    train_loss / nb_steps, 
                                                    normalized_error / nb_steps,
                                                    t_epoch - t_epoch_start))
            self.agr_train_loss.append(train_loss / nb_steps)
            self.train_agr_normalized_error.append(normalized_error / nb_steps)
            test_loss, test_normalized_error = self.test(epoch)
            self.agr_test_loss.append(test_loss)
            self.test_agr_normalized_error.append(test_normalized_error)
            scheduler.step()
            
        print('Training finished !')
        
    def test(self, epoch):
        """
        Tests the model
        """
        if self.is_multi_head:
            self.model.is_training = False
        test_loss, nb_steps, normalized_error = 0.0, 0.0, 0.0
        criterion = nn.L1Loss(reduce=False)
        for i, data in enumerate(self.test_loader):
            if self.is_multi_head :
                volume_label = Variable(data[1]['volume'].float())
                spatial_label = Variable(torch.transpose(torch.stack([data[1]['height'],
                                                                          data[1]['width'], 
                                                                          data[1]['length']]),
                                                         0, 1).float())
            elif len(self.target) > 1 and (not self.is_multi_head):
                label = Variable(torch.transpose(torch.stack([data[1]['height'],
                                                                  data[1]['width'], 
                                                                  data[1]['length'],
                                                                  data[1]['volume']]),
                                                     0, 1).float())
            else:
                label = Variable(data[1][self.target[0]].float())
            dmap = Variable(data[0]['depth_map'].float().unsqueeze(1))
            if self.cuda:
                dmap = dmap.cuda()
                if self.is_multi_head:
                    volume_label = volume_label.cuda()
                    spatial_label = spatial_label.cuda()
                    spatial_extents, volume = self.model.forward(dmap)
                else:
                    label = label.cuda()
                    volume = self.model.forward(dmap)
            # Loss computation
            if self.is_multi_head:
                spatial_loss = criterion(spatial_extents, spatial_label)
                volume_loss = criterion(volume, volume_label)
                loss = spatial_loss.mean() + volume_loss.mean()
                normalized_error += float((volume_loss / volume_label).mean())
            else:
                loss = criterion(volume, label)
                if self.model.target_size > 1:
                    normalized_error += float((loss[:, -1] / label[:, -1]).mean())
                else:
                    normalized_error += float((loss / label).mean())
            test_loss += float(loss.mean())
            
            nb_steps += 1
                
        print('epoch : {}, average test loss: {},'
              'average test error : {}'.format(epoch + 1, test_loss / nb_steps, 
                                          normalized_error / nb_steps))
        if self.is_multi_head:
            self.model.is_training = True
        return test_loss / nb_steps, normalized_error / nb_steps
    
    def save_model(self, file_name):
        file_path = 'models/' + file_name + '.pt'
        torch.save(self.model.state_dict(), file_path)
        
    def load_model(self, file_name):
        file_path = 'models/' + file_name + '.pt'
        self.model.load_state_dict(torch.load(file_path))