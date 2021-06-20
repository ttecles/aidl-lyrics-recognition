import os
import time

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer
import wandb
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from transformers import Wav2Vec2Tokenizer, AutoTokenizer

from lyre.data import DaliDataset
from lyre.model import DemucsWav2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(predicted_batch, ground_truth_batch):
    pred = predicted_batch.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acum = pred.eq(ground_truth_batch.view_as(pred)).sum().item()
    return acum

def correct_sentence(input_text):
    sentences = nltk.sent_tokenize(input_text)
    return (' '.join([s.replace(s[0], s[0].capitalize(), 1) for s in sentences]))

# def decode(id2letter, logits):
#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = wav2vec_tokenizer.decode(predicted_ids[0])
#     return correct_sentence(transcription.lower())

def convert_id_to_string(tokenizer, predicted_ids):
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids.squeeze())
    predicted_string = ''
    for token in predicted_tokens:
        if token == '<pad>':
            pass
        elif token == '|':
            predicted_string += ' '
        else:
            predicted_string += token

    return ' '.join(predicted_string.split())

def train_single_epoch(data: DataLoader, model, optimizer, criterion):
    model.train()
    train_losses = []
    for waveform, lyrics in data:
        waveform = waveform.to(device)
        optimizer.zero_grad()
        output = model(waveform)
        loss = criterion(output, lyrics)
        train_losses.append(float(loss))

        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return train_losses


def eval_single_epoch(data: DataLoader, model, criterion):
    model.eval()

    val_losses = []
    with torch.no_grad():
        for batch, (waveform, lyrics) in data:
            output = model(waveform)
            loss = criterion(output, lyrics)

            val_losses.append(float(loss))
    return val_losses


# Training
def train_model(train_data, val_data, model, optimizer, criterion, epochs):
    losses = {'train': [], 'valid': []}
    for epoch in range(epochs):
        start_time = time.time()
        train_losses = train_single_epoch(train_data, model, optimizer, criterion)
        val_losses = eval_single_epoch(val_data, model, criterion)

        # Loss average
        average_train_loss = np.mean(train_losses)
        average_valid_loss = np.mean(val_losses)
        losses['train'].append(average_train_loss)
        losses['valid'].append(average_valid_loss)

        wandb.log({"train_loss": average_train_loss})
        wandb.log({"valid_loss": average_valid_loss})

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        print(f"EPOCH: {epoch + 1},  | time in {mins} minutes, {secs} seconds")
        # print progress
        print(f'=> train loss: {average_train_loss:0.3f}  => valid loss: {average_valid_loss:0.3f}', flush=True)


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
    test_loss /= len(test_data.dataset)  # test_data.dataset is supposed to be the test split in the dataset
    test_acc = 100. * acc / len(test_data.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, acc, len(test_data.dataset),
                                                                                 test_acc))
    return test_loss, test_acc



if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    config_defaults = {
        'audio_length': 10 * 44100,
        'stride': None,
        'epochs': 5,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 0.0001,
        'optimizer': 'adam',
        'dropout': 0.5,
    }

    wandb.login(key=os.environ['WANDB_KEY'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project='demucs+wav2vec', entity='aidl-lyrics-recognition',
               config=config_defaults)
    config = wandb.config

    # Load the dataset
    if not os.path.isdir("../data"):
        raise RuntimeError("Are you in the root folder")
    dataset = DaliDataset("../data", config.audio_length, stride=config.stride)

    # Split train and val datasets
    val_len = int(len(dataset) * 0.10)
    test_len = int(len(dataset) * 0.10)
    test_dataset, valid_dataset, train_dataset = \
        random_split(dataset, [test_len, val_len, len(dataset) - val_len - test_len])

    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")


    def collate(batch: list):
        pass
    # # tokenizer.batch_decode(encoded)
    #     for b in batch:
    #         waveform, lyric = b
    #         tokenizer("AIDL", return_tensors='pt')
    #         tokenizer(["I can create some tokens".upper(), "nothing to share".upper()], return_tensors='pt',
    #                   padding=True)
    #     return source, batch_target

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)

    # Load the model
    model = DemucsWav2Vec().to(device)

    wandb.watch(model)
    criterion = torch.nn.CTCLoss().to(device)

    # Setup optimizer and LR scheduler
    # Define the optimizer
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise RuntimeError("No Optimizer specified")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    train_model(train_loader, val_loader, model, optimizer, criterion, config.epochs)

    print("Training finished")

    test_loss, test_acc = test_model(test_loader, model, criterion)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    # # Now save the artifacts of the training
    # savedir = "app/state_dict.pt"
    # print(f"Saving checkpoint to {savedir}...")
    # # We can save everything we will need later in the checkpoint.
    # checkpoint = {
    #     "model_state_dict": model.cpu().state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    # }
    # torch.save(checkpoint, savedir)
