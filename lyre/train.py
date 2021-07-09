import argparse
import os
import pathlib
import time

import DALI as dali_code
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from lyre.data import DaliDataset, DEFAULT_SAMPLE_RATE
from lyre.model import DemucsWav2Vec

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32


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


def eval_single_epoch(data: DataLoader, model, criterion, accelerator):
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


# def test_model(test_data, model, criterion):
#     model.eval()
#     test_loss = 0
#     acc = 0
#     with torch.no_grad():
#         for waveform, lyrics in test_data:
#             output = model(test_data)
#
#             test_loss += criterion(output, lyrics)
#             acc += accuracy(output, lyrics)
#             print(test_loss, acc)
#     test_loss /= len(test_data.dataset)  # test_data.dataset is supposed to be the test split in the dataset
#     test_acc = 100. * acc / len(test_data.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, acc, len(test_data.dataset),
#                                                                                  test_acc))
#     return test_loss, test_acc


def save_model(model, optimizer, epoch, loss, folder):
    print(f"Saving checkpoint to {folder}/model.pt...")
    # We can save everything we will need later in the checkpoint.
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.cpu().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, folder + "/model.pt")


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="train")
    parser.add_argument("DALI_DATA_PATH", type=pathlib.Path)
    parser.add_argument("DALI_AUDIO_PATH", type=pathlib.Path)
    parser.add_argument("--dali-gt-file", type=pathlib.Path)
    parser.add_argument("--blacklist-file", type=pathlib.Path)
    parser.add_argument("--audio-length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    # parser.add_argument("--dropout", type=float, default=0.5)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    group.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--model-folder", help="if specified, model will be saved in every epoch into the folder")
    parser.add_argument("--load-model", action="store_true", help="loads the model before training")
    parser.add_argument("--freeze-demucs", action="store_true", help="does not train demucs")

    args = parser.parse_args()

    batch_size = args.batch
    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if args.batch > MAX_GPU_BATCH_SIZE:
        gradient_accumulation_steps = args.batch // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    config = {
        'audio_length': args.audio_length,
        'stride': args.stride,
        'epochs': args.epochs,
        'batch_size': args.batch,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        # 'dropout': args.dropout,
        'workers': args.workers,
        'freeze_demucs': args.freeze_demucs
    }

    # WANDB configuration
    if 'WANDB_KEY' in os.environ:
        wandb.login(key=os.environ['WANDB_KEY'])
        os.environ["WANDB_SILENT"] = "true"
    wandb.init(project='demucs+wav2vec', entity='aidl-lyrics-recognition',
               config=config)
    config = wandb.config

    # User input validation and transformation
    if args.blacklist_file:
        with open(args.blacklist_file) as f:
            blacklist = f.read().splitlines()
    else:
        blacklist = []

    audio_length = args.audio_length * DEFAULT_SAMPLE_RATE
    if args.stride:
        stride = args.stride * DEFAULT_SAMPLE_RATE
    else:
        stride = args.stride

    if args.dali_gt_file:
        gt_file = str(args.dali_gt_file.resolve(strict=True))
    else:
        gt_file = ''

    print("Loading DALI dataset...")
    dali_data = dali_code.get_the_DALI_dataset(str(args.DALI_DATA_PATH.resolve(strict=True)),
                                               gt_file=gt_file,
                                               skip=blacklist,
                                               keep=[])

    print("Preparing Datasets...")
    test_dataset = DaliDataset(dali_data, dali_audio_path=args.DALI_AUDIO_PATH.resolve(strict=True),
                               length=audio_length, stride=stride, ncc=(.94, None), workers=args.workers)
    print(f"Test DaliDataset: {len(test_dataset)} chunks")
    validation_dataset = DaliDataset(dali_data, dali_audio_path=args.DALI_AUDIO_PATH.resolve(strict=True),
                                     length=audio_length, stride=stride, ncc=(.925, .94), workers=args.workers)
    print(f"Validation DaliDataset: {len(validation_dataset)} chunks")
    train_dataset = DaliDataset(dali_data, dali_audio_path=args.DALI_AUDIO_PATH.resolve(strict=True),
                                length=audio_length, stride=stride, ncc=(.8, .925), workers=args.workers)
    print(f"Train DaliDataset: {len(train_dataset)} chunks")

    tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    def collate(batch: list):
        # tokenizer.batch_decode(encoded)
        waveforms, lyrics = zip(*batch)
        lyrics_ids = tokenizer(lyrics, return_tensors='pt', padding=True)['input_ids']
        return torch.stack(waveforms), lyrics_ids

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
                             num_workers=args.workers)

    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)
    # Load the model
    model = DemucsWav2Vec()
    if args.load_model:
        checkpoint = torch.load(args.model_folder)
        model.load_state_dict(checkpoint["model_state_dict"])

    wandb.watch(model)

    # Setup optimizer and LR scheduler
    # Define the optimizer
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise RuntimeError("No Optimizer specified")

    if args.load_model:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    criterion = torch.nn.CTCLoss()

    model, optimizer, train_loader, val_loader, test_loader, criterion = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, criterion
    )

    accelerator.print("Start Training with device", str(accelerator.device))
    losses = {'train': [], 'valid': []}
    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        train_losses = []
        for idx, batch in enumerate(train_loader):
            waveform, lyrics = batch
            logits, voice = model(waveform)
            batch_size, input_lengths, classes = logits.size()
            _, target_lengths = lyrics.size()
            log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = criterion(log_prob, lyrics,
                             input_lengths=torch.full(size=(batch_size,), fill_value=input_lengths, dtype=torch.short),
                             target_lengths=torch.full(size=(batch_size,), fill_value=target_lengths,
                                                       dtype=torch.short))
            train_losses.append(loss.item())

            accelerator.backward(loss)

            optimizer.step()
            if scheduler:
                scheduler.step(loss)

            optimizer.zero_grad()

            predicted_ids = torch.argmax(logits, dim=-1)
        wandb_data = {"batch_train_loss": loss.item(),
                   "input": wandb.Audio(waveform[0].mean(0).detach().numpy(),
                                        sample_rate=model.demucs.samplerate),
                   "voice": wandb.Audio(voice[0].detach().numpy(), sample_rate=model.sr_wav2vec),
                   "predictions": wandb.Html(f"""<table style="width:100%">
                   <tr><th>Epoch</th> <th>Batch ID</th> <th>Lyric</th> <th>Predicted</th> </tr>
                   <tr><td>{epoch}</td>
                   <td>{idx}</td>
                   <td>{tokenizer.decode(lyrics[0])}</td>
                   <td>{tokenizer.batch_decode(predicted_ids)[0]}</td></tr>
                   </table>"""), }


        model.eval()
        val_losses = []

        for waveform, lyrics in val_loader:
            with torch.no_grad():
                logits, voice = model(waveform)
                batch_size, input_lengths, classes = logits.size()
                _, target_lengths = lyrics.size()
                log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                loss = criterion(log_prob, lyrics,
                                 input_lengths=torch.full(size=(batch_size,), fill_value=input_lengths,
                                                          dtype=torch.short),
                                 target_lengths=torch.full(size=(batch_size,), fill_value=target_lengths,
                                                           dtype=torch.short))
                val_losses.append(loss.item())

        # Loss average
        average_train_loss = np.mean(train_losses)
        average_valid_loss = np.mean(val_losses)
        losses['train'].append(average_train_loss)
        losses['valid'].append(average_valid_loss)

        wandb.log({"train_loss": average_train_loss, "valid_loss": average_valid_loss, **wandb_data})

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        accelerator.print(f"EPOCH: {epoch + 1},  | time in {mins} minutes, {secs} seconds")
        # print progress
        accelerator.print(f'=> train loss: {average_train_loss:0.3f}  => valid loss: {average_valid_loss:0.3f}')

        if args.model_folder:
            save_model(model, optimizer, epoch, losses['train'][-1], args.model_folder)
    accelerator.print("Training finished")

    # test_loss, test_acc = test_model(test_loader, model, criterion)
    # accelerator.print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
    # wandb.log({"test_loss": test_loss, "test_acc": test_acc, "lyrics": table})


if __name__ == "__main__":
    main()
