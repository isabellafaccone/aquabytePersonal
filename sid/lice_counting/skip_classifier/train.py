import time
import pytz
from datetime import datetime
import copy
import numpy as np
import argparse
import os
import json

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim

from loader import TRANSFORMS, CLASS_COUNTS, NUM_EX, get_dataloader

CHECKPOINT_PATH = 'checkpoints'

def get_model(class_names, device, savename):
    print('Building model object...')
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)
    if state_dict_path is not None:
        print(f'Loading from {state_dict_path}')
        model_ft.load_state_dict(torch.load(state_dict_path))
    return model_ft

def get_metrics(preds: torch.Tensor, labels: torch.Tensor, last_batch, class_idxs=[0, 1]):
    """
    preds: 1-D tensor
    labels: 1-D tensor
    """
    report = dict()
    for lab in labels:
        correct = (preds == labels)
        report[lab] = {
            'precision' : torch.mean(correct[preds == lab]).float(),
            'recall' : torch.mean(correct[labels == lab]).float(),
            'acc' : torch.mean(correct).float()
        }
        # if last_batch:
        #     for m in report[lab]:
        #         report[lab][m] = preds.size(0) /
    return report

def train_model(dataloaders, model_ft, class_names, savename, bsz, device=0, num_epochs=25, phases=['val']):

    # Weight classes in cross entropy loss function based on relative frequency
    print('Preparing training job...')
    if class_names == ['QA', 'SKIPPED_ANN']:
        weight = torch.Tensor([CLASS_COUNTS['SKIPPED_ANN']/CLASS_COUNTS['QA'], 1]).cuda()
    elif class_names == ['SKIPPED_ANN', 'QA']:
        weight = torch.Tensor([1, CLASS_COUNTS['SKIPPED_ANN']/CLASS_COUNTS['QA']]).cuda()
    else:
        assert False

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft.zero_grad()

    since = time.time()

    best_model_ft_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model_ft.train()  # Set model_ft to training mode
            else:
                model_ft.eval()   # Set model_ft to evaluate mode

            running_loss = 0.0
            running_accs = {lab: [] for lab in class_names + ['macro']}

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                if i > 10:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                last_batch = i == ((NUM_EX // bsz) - 1)
                start_cal = time()
                running_metrics.append(get_metrics(preds, labels.data, last_batch))
                finish_cal = time() - start_cal
                print(f'Calculated metrics in {finish_cal}')
            # if phase == 'train':
            #     exp_lr_scheduler.step()

            epoch_loss = running_loss / {'train': 80000, 'val': 10000}[phase]
            epoch_metrics = dict()
            for i, lab in enumerate(class_names):
                epoch_acc[lab] = np.mean(running_accs[lab])

            epoch_acc['macro'] = np.mean(list(epoch_acc.values()))
            epoch_acc['micro'] = np.mean(running_accs['micro'])

            print('{} Loss: {:.4f} Acc: {}'.format(
                phase, epoch_loss, epoch_acc))

            # Save results
            save_dir = f'checkpoints/{savename}/epoch_{epoch}/{phase}/'
            os.makedirs(save_dir, exist_ok=True)

            torch.save(model_ft.state_dict(), os.path.join(save_dir, 'model.pt'))
            json.dump({'acc': epoch_acc, 'loss': epoch_loss}, open(os.path.join(save_dir, 'metrics.json'), 'w'))

            # deep copy the model_ft
            if phase == 'val' and epoch_acc['macro'] > best_acc:
                best_acc = epoch_acc['macro']
                best_model_ft_wts = copy.deepcopy(model_ft.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model_ft weights
    model_ft.load_state_dict(best_model_ft_wts)
    return model_ft

def run(transform, savename, bsz, split_size, device):
    torch.cuda.set_device(device)
    dataloaders, class_names = get_dataloader(transform, bsz, split_size)
    model = get_model(class_names, device, savename)
    trained_model = train_model(dataloaders, model, class_names, savename, device)
    return trained_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Skip classifier.')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--transform', type=str)
    parser.add_argument('--savename', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--split_size', type=float, default=0.8)

    args = parser.parse_args()
    args.savename = args.savename + '__' + datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y-%m-%d__%H-%M-%S")
    run(
        TRANSFORMS[args.transform],
        args.savename,
        args.batch_size,
        args.split_size,
        device=args.device
    )
