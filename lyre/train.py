import argparse
import os
import pathlib
import time

import DALI as dali_code
import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lyre.data import DaliDataset
from lyre.model import DemucsWav2Vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def accuracy(predicted_batch, ground_truth_batch):
    pred = predicted_batch.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acum = pred.eq(ground_truth_batch.view_as(pred)).sum().item()
    return acum

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
        batch, input_lengths, classes = output.size()
        _, target_lengths = lyrics.size()
        output = output.permute(1, 0, 2)
        loss = criterion(output, lyrics,
                         input_lengths=torch.full(size=(batch,), fill_value=input_lengths, dtype=torch.long),
                         target_lengths=torch.full(size=(batch,), fill_value=target_lengths, dtype=torch.long))
        train_losses.append(float(loss))

        loss.backward()
        optimizer.step()
        wandb.log({"batch loss": loss.item()})
    return train_losses


def eval_single_epoch(data: DataLoader, model, criterion):
    model.eval()

    val_losses = []
    with torch.no_grad():
        for waveform, lyrics in data:
            output = model(waveform)
            batch, input_lengths, classes = output.size()
            _, target_lengths = lyrics.size()
            loss = criterion(output, lyrics,
                             input_lengths=torch.full(size=(batch,), fill_value=input_lengths, dtype=torch.long),
                             target_lengths=torch.full(size=(batch,), fill_value=target_lengths, dtype=torch.long)
                             )

            val_losses.append(float(loss))
    return val_losses


# Training
def train_model(train_data, val_data, model, optimizer, criterion, epochs, vocab_size):
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
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("DALI_DATA_PATH", type=pathlib.Path)
    parser.add_argument("DALI_AUDIO_PATH", type=pathlib.Path)
    parser.add_argument("--dali-gt-file", type=pathlib.Path)
    parser.add_argument("--blacklist-file", type=pathlib.Path)
    parser.add_argument("--audio-length", type=int, default=10)
    parser.add_argument("--stride")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--workers", type=int, default=0)


    namespace = parser.parse_args()

    config = {
        'audio_length': namespace.audio_length,
        'stride': namespace.stride,
        'epochs': namespace.epochs,
        'batch_size': namespace.batch,
        'learning_rate': namespace.lr,
        'weight_decay': namespace.weight_decay,
        'optimizer': namespace.optimizer,
        'dropout': namespace.dropout,
        'workers': namespace.workers
    }

    if 'WANDB_KEY' in os.environ:
        wandb.login(key=os.environ['WANDB_KEY'])
        os.environ["WANDB_SILENT"] = "true"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project='demucs+wav2vec', entity='aidl-lyrics-recognition',
               config=config)
    config = wandb.config
    # Load the dataset

    if namespace.blacklist_file:
        with open(namespace.blacklist_file) as f:
            blacklist = f.read().splitlines()
    else:
        blacklist = []

    if namespace.cpu:
        device = torch.device("cpu")

    print("Loading DALI dataset...")
    dali_data = dali_code.get_the_DALI_dataset(str(namespace.DALI_DATA_PATH.resolve(strict=True)),
                                               gt_file=str(namespace.dali_gt_file.resolve(strict=True)),
                                               skip=blacklist,
                                               keep=[])

    print("Preparing Datasets...")
    test_dataset = DaliDataset(dali_data, dali_audio_path=namespace.DALI_AUDIO_PATH.resolve(strict=True),
                               length=config.audio_length,
                               stride=config.stride, ncc=(.94, None), workers=namespace.workers)
    print(f"Test DaliDataset: {len(test_dataset)} chunks")
    validation_dataset = DaliDataset(dali_data, dali_audio_path=namespace.DALI_AUDIO_PATH.resolve(strict=True),
                                     length=config.audio_length,
                                     stride=config.stride, ncc=(.925, .94), workers=namespace.workers)
    print(f"Validation DaliDataset: {len(validation_dataset)} chunks")
    train_dataset = DaliDataset(dali_data, dali_audio_path=namespace.DALI_AUDIO_PATH.resolve(strict=True),
                                length=config.audio_length,
                                stride=config.stride, ncc=(.8, .925), workers=namespace.workers)
    print(f"Train DaliDataset: {len(train_dataset)} chunks")

    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")


    def collate(batch: list):
        # tokenizer.batch_decode(encoded)
        waveforms, lyrics = zip(*batch)
        lyrics_ids = tokenizer(lyrics, return_tensors='pt', padding=True)['input_ids']
        return torch.stack(waveforms), lyrics_ids


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=namespace.workers)
    val_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=namespace.workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate,
                             num_workers=namespace.workers)

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

    print("Start Training with device", str(device))

    train_model(train_data=train_loader, val_data=val_loader, model=model, optimizer=optimizer, criterion=criterion,
                epochs=config.epochs, vocab_size=tokenizer.vocab_size)

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
