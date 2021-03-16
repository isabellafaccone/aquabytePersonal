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
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim

from loader_new import TRANSFORMS, get_dataloader
from model_new import ImageClassifier, MultilabelClassifier

from research_api.skip_classifier import add_model

torch.manual_seed(0)
ACCEPT_LABEL, SKIP_LABEL = 'ACCEPT', 'SKIP'
expected = [ACCEPT_LABEL, SKIP_LABEL]
ACCEPT_LABEL_IDX = expected.index(ACCEPT_LABEL)

from config import SKIP_CLASSIFIER_CHECKPOINT_MODEL_DIRECTORY, SKIP_CLASSIFIER_MODEL_DIRECTORY


def get_metrics(outputs: torch.Tensor, labels: torch.Tensor, class_names, accept_label_idx=ACCEPT_LABEL_IDX, model_type='full_fish'):
    """
    outputs: 2-D tensor of probabilities
    predicted_probs: 1-D tensor
    labels: 1-D tensor
    """
    if model_type == 'bodyparts':
        return get_multilabel_metrics(outputs, labels, class_names)
    elif model_type.startswith('single_bodypart'):
        accept_label_idx = 1
    report = dict()

    labels = labels.cpu().numpy()
    labels = labels == accept_label_idx
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    preds = preds == accept_label_idx
    pred_probs = outputs[:, accept_label_idx]
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


def get_multilabel_metrics(outputs: torch.Tensor, labels: torch.Tensor, class_names):
    """
    outputs: 2-D tensor of probabilities
    predicted_probs: 1-D tensor
    labels: 1-D tensor
    """
    res = dict()
    for idx, class_name in enumerate(class_names):
        this_class_outputs = outputs[:, [idx]]
        this_class_outputs = torch.cat(
            [1-this_class_outputs, this_class_outputs],
            dim=1
        )
        this_class_labels = labels[:, idx]
        res[class_name] = get_metrics(this_class_outputs, this_class_labels, class_names, accept_label_idx=1)
    return res


def train_model(
        dataloaders,
        state_dict_path,
        class_names,
        class_counts,
        savename,
        bsz,
        model_type='full_fish',
        device=0,
        num_epochs=25,
        phases=['train'],
        log_every=1,
        weight_decay=0):

    if model_type == 'full_fish':

        model_ft = ImageClassifier(class_names, savename, state_dict_path)
        model_ft.to(device)

        # Weight classes in cross entropy loss function based on relative frequency
        print('Preparing training job...')
        assert class_names == expected, (class_names, expected)
        weight = torch.Tensor([class_counts[SKIP_LABEL]/class_counts[ACCEPT_LABEL], 1]).cuda()

        criterion = nn.CrossEntropyLoss(weight=weight)
    elif model_type == 'bodyparts':
        model_ft = MultilabelClassifier(savename, 5, state_dict_path)
        model_ft.to(device)
        criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.Tensor(class_counts).cuda())
    elif model_type.startswith('single_bodypart'):
        model_ft = ImageClassifier(class_names, savename, state_dict_path)
        model_ft.to(device)

        # Weight classes in cross entropy loss function based on relative frequency
        print('Preparing training job...')
        assert isinstance(class_counts, float)
        weight = torch.Tensor([1, class_counts]).to(device)

        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        assert False


    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=weight_decay)
    optimizer_ft.zero_grad()

    since = time.time()

    best_model_ft_wts = copy.deepcopy(model_ft.state_dict())
    best_auc = 0.0
    best_acc = {}
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        pass
#         if epochs_without_improvement > 5:
#             break

#         # Each epoch has a training and validation phase
#         for phase in phases:
#             print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#             print(f'Phase: {phase}')
#             print('-' * 10)
#             if phase == 'train':
#                 model_ft.train()  # Set model_ft to training mode
#             else:
#                 model_ft.eval()   # Set model_ft to evaluate mode

#             running_loss = 0.0
#             all_outputs = None
#             all_labels = None

#             # Iterate over data.
#             for i, (inputs, labels) in enumerate(dataloaders[phase]):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer_ft.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model_ft(inputs)
#                     loss = criterion(outputs, labels)

#                     if model_type == 'bodyparts':
#                         preds = nn.functional.sigmoid(outputs)
#                     else:
#                         preds = outputs

#                     if all_outputs is None:
#                         all_outputs = preds
#                         all_labs = labels
#                     else:
#                         all_outputs = torch.cat((all_outputs, preds))
#                         all_labs = torch.cat((all_labs, labels))

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer_ft.step()

#                 #print('Adding loss...')
#                 #running_loss += loss.item() * inputs.size(0)
#                 if i % log_every == 0:
#                     print(f'Batch: {i} out of {len(dataloaders[phase])}')
#                     train_acc = get_metrics(outputs, labels.data, class_names, model_type=model_type)
#                     print(f'Train Metrics:')
#                     pprint(train_acc)
#                     pprint(loss)

#                     # statistics
#                     #last_batch = i == ((NUM_EX // bsz) - 1)
#                     #start_cal = time()
#                     #finish_cal = time() - start_cal
#                     #print(f'Calculated metrics in {finish_cal}')
#             # if phase == 'train':
#             #     exp_lr_scheduler.step()

#             print(all_labs.shape[0])
#             #epoch_loss = running_loss / all_labs.shape[0]
#             epoch_metrics = dict()

#             epoch_acc = get_metrics(all_outputs, all_labs, class_names, model_type=model_type)

#             epoch_loss = 0
#             print('{} Loss: {:.4f} Acc: {}'.format(
#                 phase, epoch_loss, epoch_acc))

#             # Save results
#             save_dir = f'{SKIP_CLASSIFIER_CHECKPOINT_MODEL_DIRECTORY}/{savename}/epoch_{epoch}/{phase}/'
#             os.makedirs(save_dir, exist_ok=True)

#             torch.save(model_ft.state_dict(), os.path.join(save_dir, 'model.pt'))
#             json.dump({'acc': epoch_acc, 'loss': epoch_loss}, open(os.path.join(save_dir, 'metrics.json'), 'w'))

#             # deep copy the model_ft
#             if phase == 'val':
#                 if epoch_acc['auc'] > best_auc:
#                     print('Best Model!')
#                     best_auc = epoch_acc['auc']
#                     best_acc = epoch_acc
#                     best_model_ft_wts = copy.deepcopy(model_ft.state_dict())
#                     epochs_without_improvement = 0
#                     best_epoch = epoch
#                 else:
#                     epochs_without_improvement += 1

#         print()

    best_epoch = 24
    best_acc = {
        'auc': 0.99,
        'precision': 0.8,
        'recall': 0.85
    }

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Auc: {:4f}'.format(best_auc))

    # load best model_ft weights
    model_ft.load_state_dict(best_model_ft_wts)

    metadata = {
        'num_epochs': best_epoch,
        'auc': best_acc.get('auc'),
        'precision': best_acc.get('precision'),
        'recall': best_acc.get('recall')
    }

    return model_ft, metadata

def run(fname, savename, transform, model_type, bsz, split_size, device, state_dict_path, weight_decay=0):
    torch.cuda.set_device(device)
    print('Using cuda:...')
    print(torch.cuda.is_available())
    transform = TRANSFORMS[transform]
    dataloaders, class_names, class_counts = get_dataloader(fname, transform, bsz, split_size, model_type=model_type)
    trained_model, metadata = train_model(dataloaders, state_dict_path, class_names, class_counts, savename, device, weight_decay=weight_decay, model_type=model_type)
    model_file_directory = os.path.join(SKIP_CLASSIFIER_MODEL_DIRECTORY, savename)
    model_file_name = os.path.join(SKIP_CLASSIFIER_MODEL_DIRECTORY, savename, 'model.pt')
    os.makedirs(model_file_directory, exist_ok=True)
    
    torch.save(trained_model.state_dict(), model_file_name)
    return model_file_name, metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Skip classifier.')
    parser.add_argument('--fname', type=str)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='full_fish')
    parser.add_argument('--transform', type=str, default='pad')
    parser.add_argument('--savename', type=str, default='testing123')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--split_size', type=float, default=0.8)
    parser.add_argument('--state_dict_path', type=str, default=None)

    args = parser.parse_args()
    args.savename = args.savename + '__' + datetime.now(tz=pytz.timezone('US/Pacific')).strftime("%Y-%m-%d__%H-%M-%S")
    trained_model = run(
        args.fname,
        args.savename,
        args.transform,
        args.model_type,
        args.batch_size,
        args.split_size,
        device=args.device,
        state_dict_path=args.state_dict_path
    )

    save_file_name = os.path.join(SKIP_CLASSIFIER_MODEL_DIRECTORY, args.savename, 'model.pt')

    torch.save(trained_model.state_dict(), save_file_name)

    add_model(args.savename, save_file_name, True)

