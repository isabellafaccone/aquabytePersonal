import time
import pytz
from datetime import datetime
import copy
import numpy as np
import argparse
import os
import json
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

from loader import TRANSFORMS, get_dataloader
from model import ImageClassifier

torch.manual_seed(0)
ACCEPT_LABEL, SKIP_LABEL = 'ACCEPT', 'SKIP'
expected = [ACCEPT_LABEL, SKIP_LABEL]
ACCEPT_LABEL_IDX = expected.index(ACCEPT_LABEL)
CHECKPOINT_PATH = '/root/data/sid/skip_classifier_checkpoints'


def get_metrics(outputs: torch.Tensor, labels: torch.Tensor, class_idxs=[0, 1]):
    """
    outputs: 2-D tensor of probabilities
    predicted_probs: 1-D tensor
    labels: 1-D tensor
    """
    report = dict()

    labels = labels.cpu().numpy()
    labels = labels == ACCEPT_LABEL_IDX
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    preds = preds == ACCEPT_LABEL_IDX
    pred_probs = outputs[:, ACCEPT_LABEL_IDX]
    pred_probs = pred_probs.detach().cpu().numpy()
    if preds.sum():
        precision = precision_score(labels, preds)
    else:
        print('No pred positives')
        precision = 1
    if labels.sum():
        recall = recall_score(labels, preds)
    else:
        print('No true positives')
        recall = 1
    try:
        auc = roc_auc_score(labels, pred_probs)
    except:
        auc = 0
    return {
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

def train_model(dataloaders, model_ft, class_names, class_counts, savename, bsz, device=0, num_epochs=25, phases=['train', 'val'], log_every=10):

    # Weight classes in cross entropy loss function based on relative frequency
    print('Preparing training job...')
    assert class_names == expected, (class_names, expected)
    weight = torch.Tensor([class_counts[SKIP_LABEL]/class_counts[ACCEPT_LABEL], 1]).cuda()

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft.zero_grad()

    since = time.time()

    best_model_ft_wts = copy.deepcopy(model_ft.state_dict())
    best_auc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        if epochs_without_improvement > 5:
            break

        # Each epoch has a training and validation phase
        for phase in phases:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print(f'Phase: {phase}')
            print('-' * 10)
            if phase == 'train':
                model_ft.train()  # Set model_ft to training mode
            else:
                model_ft.eval()   # Set model_ft to evaluate mode

            running_loss = 0.0
            all_outputs = None
            all_labels = None

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)
                    loss = criterion(outputs, labels)

                    if all_outputs is None:
                        all_outputs = outputs
                        all_labs = labels
                    else:
                        all_outputs = torch.cat((all_outputs, outputs))
                        all_labs = torch.cat((all_labs, labels))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_ft.step()

                running_loss += loss.item() * inputs.size(0)
                if i % log_every == 0:
                    print(f'Batch: {i}')
                    train_acc = get_metrics(outputs, labels.data)
                    print(f'Metrics: {train_acc}')
                    # statistics
                    #last_batch = i == ((NUM_EX // bsz) - 1)
                    #start_cal = time()
                    #finish_cal = time() - start_cal
                    #print(f'Calculated metrics in {finish_cal}')
            # if phase == 'train':
            #     exp_lr_scheduler.step()

            print(all_labs.shape[0])
            epoch_loss = running_loss / all_labs.shape[0]
            epoch_metrics = dict()

            epoch_acc = get_metrics(all_outputs, all_labs)

            print('{} Loss: {:.4f} Acc: {}'.format(
                phase, epoch_loss, epoch_acc))

            # Save results
            save_dir = f'{CHECKPOINT_PATH}/{savename}/epoch_{epoch}/{phase}/'
            os.makedirs(save_dir, exist_ok=True)

            torch.save(model_ft.state_dict(), os.path.join(save_dir, 'model.pt'))
            json.dump({'acc': epoch_acc, 'loss': epoch_loss}, open(os.path.join(save_dir, 'metrics.json'), 'w'))

            # deep copy the model_ft
            if phase == 'val':
                if epoch_acc['auc'] > best_auc:
                    print('Best Model!')
                    best_auc = epoch_acc['auc']
                    best_model_ft_wts = copy.deepcopy(model_ft.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Auc: {:4f}'.format(best_auc))

    # load best model_ft weights
    model_ft.load_state_dict(best_model_ft_wts)
    return model_ft

def run(fname, savename, transform, bsz, split_size, device, state_dict_path):
    torch.cuda.set_device(device)
    transform = TRANSFORMS[transform]
    dataloaders, class_names, class_counts = get_dataloader(fname, transform, bsz, split_size)
    model = ImageClassifier(class_names, savename, state_dict_path)
    model.to(device)
    trained_model = train_model(dataloaders, model, class_names, class_counts, savename, device)
    return trained_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Skip classifier.')
    parser.add_argument('--fname', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--transform', type=str, default='pad')
    parser.add_argument('--savename', type=str, default='testing123')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--split_size', type=float, default=0.8)
    parser.add_argument('--state_dict_path', type=str, default=None)

    args = parser.parse_args()
    args.savename = args.savename + '__' + datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y-%m-%d__%H-%M-%S")
    run(
        args.fname,
        args.savename,
        args.transform,
        args.batch_size,
        args.split_size,
        device=args.device,
        state_dict_path=args.state_dict_path
    )
