import numpy as np
import torch
import torch.nn as nn

from .trainUtils import accuracy


# Train single epoch
def train_single_epoch(train_data, model, optimizer, criterion, epoch):
    model.train()

    train_losses = []

    for batch, (waveform, lyrics) in train_data:

        optimizer.zero_grad()
        output = model(waveform)
        loss = criterion(output, lyrics)

        train_losses.append(float(loss))

        loss.backward()
        optimizer.step()

    return train_losses


# Eval single epoch
def eval_single_epoch(val_data, model, criterion, epoch):
    model.eval()

    val_losses = []

    with torch.no_grad():
        for batch, (waveform, lyrics) in val_data:

            output = model(waveform)
            loss = criterion(output, lyrics)

            val_losses.append(float(loss))

    return val_losses


# Training
def train_model(model, optimizer, criterion, epochs):

    losses = {'train': [], 'valid': []}
    for epoch in range(epochs):
        train_losses = train_single_epoch(train_data, model, optimizer, criterion, epoch)
        val_losses = eval_single_epoch(val_data, model, criterion, epoch)

        # Loss average
        average_train_loss = np.mean(train_losses)
        average_valid_loss = np.mean(val_losses)
        losses['train'].append(average_train_loss)
        losses['valid'].append(average_valid_loss)

        # print progress
        print(f'Epoch: {epoch}  '
        f'train loss: {average_train_loss:0.3f}  '
        f'valid loss: {average_valid_loss:0.3f}', flush=True)


# Testing
def test_model(test_data, model, criterion):

    model.eval()

    test_loss = 0
    acc = 0
    with torch.no_grad():
        for batch, (waveform, lyrics) in test_data:

            output = model(test_data)

            test_loss += criterion(output, lyrics)
            acc += accuracy(output, lyrics)
            print(test_loss, acc)
    test_loss /= len(test_data.dataset) # test_data.dataset is supposed to be the test split in the dataset
    test_acc = 100. * acc / len(test_data.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_data.dataset), test_acc,
    ))
    return test_loss, test_acc