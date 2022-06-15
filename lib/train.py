import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm_notebook
import numpy as np
import time

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def train_epoch(model, optimizer, criterion, train_dataloader, device, optim):
    model.train()
    loss_epoch = []
    for batch in train_dataloader:
        if optim == 'LBFGS':
            def closure():
                optimizer.zero_grad()
                prediction = model(batch[0].to(device))
                loss = criterion(prediction, batch[1].to(device))
                loss.backward()
                return loss
            prediction = model(batch[0].to(device))
            loss = criterion(prediction, batch[1].to(device))
            optimizer.step(closure)
            
        elif optim == 'AdaHessian':
            optimizer.zero_grad()
            prediction = model(batch[0].to(device))
            loss = criterion(prediction, batch[1].to(device))
            loss.backward(create_graph=True)
            optimizer.step()

        else:
            optimizer.zero_grad()
            prediction = model(batch[0].to(device))
            loss = criterion(prediction, batch[1].to(device))
            loss.backward()
            optimizer.step()
            
        loss_epoch.append(loss.item())
        
    return np.mean(loss_epoch)

def compute_scores(predictions, targets):
    targets = np.hstack(targets)
    predictions = np.concatenate(predictions, axis = 0)
    predicted_labels = np.argmax(predictions, axis = 1)
    accuracy = accuracy_score(targets, predicted_labels, )
    f1 = f1_score(targets, predicted_labels, average='macro')
    precision = precision_score(targets, predicted_labels, average='macro')
    recall = recall_score(targets, predicted_labels, average='macro')
    
    return accuracy, f1, precision, recall


@torch.no_grad()
def valid_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    valid_epoch = []
    predictions = []
    targets = []
    for batch in valid_dataloader:
        prediction = model(batch[0].to(device))
        loss = criterion(prediction, batch[1].to(device))
        valid_epoch.append(loss.item())
        predictions.append(prediction.cpu().detach().numpy())
        targets.append(batch[1].cpu().detach().numpy())
    accuracy, f1, precision, recall = compute_scores(predictions, targets)
    
    return np.mean(valid_epoch), accuracy, f1, precision, recall

def train(model, optimizer, criterion, train_dataloader, 
          valid_dataloader, device, epochs = 100,
          optim="Adam", verbose=False):
    
    results = {'train': [], 'valid': [], 'accuracy': [], 
               'f1': [], 'precision': [], 'recall': [], 'time': []}
    
    for epoch in tqdm_notebook(range(epochs)):
        start = time.time()
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device, optim)
        end = time.time()
        valid_loss, accuracy, f1, precision, recall = valid_epoch(model, criterion, valid_dataloader, device)
        if verbose:
            print(f'Epoch: {epoch}, train_loss: {train_loss},'+
                  f'valid_loss: {valid_loss}, f1: {f1}, time: {end - start}')
        results['train'].append(train_loss)
        results['valid'].append(valid_loss)
        results['accuracy'].append(accuracy)
        results['f1'].append(f1)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['time'].append(end - start)
        
    return results