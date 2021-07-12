import argparse
import os
import pathlib
import signal
import tempfile
import time
import typing as t

import DALI as dali_code
import ctcdecode
import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

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


def save_model(model, optimizer: optim.Optimizer, folder: pathlib.Path, train_loss=None, val_loss=None, epoch=None,
               accelerator: Accelerator = None):
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except:
        print(f"failed creating {folder}. Using temp dir")
        folder = pathlib.Path(tempfile.gettempdir())

    filename = (folder / f"model_{int(time.time())}.pt").resolve()

    if accelerator:
        save_func = accelerator.save
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_optimizer = accelerator.unwrap_model(optimizer)
        model_state_dict = unwrapped_model.state_dict()
        optimizer_state_dict = unwrapped_optimizer.state_dict()
    else:
        save_func = torch.save
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

    try:
        save_func({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict
        }, str(filename))
    except PermissionError:
        try:
            filename = pathlib.Path(tempfile.gettempdir()) / filename.name
            print(f"Failed saving model. Trying into {filename}")
            save_func({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict
            }, str(filename))
        except PermissionError:
            print(f"Unable to save model")
        else:
            print(f"Saved model in {filename}")
    else:
        print(f"Saved model in {filename}")


def load_model(file: t.Union[str, pathlib.Path], model=None, optimizer=None):
    # First load into memory the variables that we will need to predict
    checkpoint = torch.load(file)

    model = model or DemucsWav2Vec()
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_stage_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(prog="train")
    data = parser.add_argument_group('data arguments', 'data input information')
    data.add_argument("--data-path", type=pathlib.Path, help="Path holding all the data.", default='./data')
    data.add_argument("--dali-gt-file", type=pathlib.Path, help="DALI ground truth file.")
    data.add_argument("--blacklist-file", type=pathlib.Path,
                      help="All the identifiers listed in the file will be skipped from being loaded.")
    data.add_argument("--lm", type=pathlib.Path,
                      help="Trained Language Model file. if not specified it will try to find it in ./data/text.arpa")

    train_config = parser.add_argument_group('train config arguments', 'configuration of the training.')
    train_config.add_argument("--ncc", type=float, default=0,
                              help="Train only with files with NCC score bigger than NCC.")
    train_config.add_argument("--train-split", type=float, default=0.7,
                              help="Train proportion. Requires --ncc to be specified.")
    train_config.add_argument("--audio-length", type=int, default=5,
                              help="Audio length in seconds to pass to the model.")
    train_config.add_argument("--stride", type=int, default=None, help="Stride used for spliting the audio songs.")
    train_config.add_argument("--epochs", type=int, default=15, help="Number of epochs during training.")
    train_config.add_argument("--batch", type=int, default=8, help="Batch size.")
    train_config.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Type of optimizer.")
    train_config.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    train_config.add_argument("--wd", type=float, default=1e-4, help="Optimizer weight decay.")
    group = train_config.add_mutually_exclusive_group()
    group.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    group.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    train_config.add_argument("--workers", type=int, default=0,
                              help="Number of workers used for processing chunks, DataLoader and decoder.")

    model_config = parser.add_argument_group('model config arguments', 'configuration of the model')
    model_config.add_argument("--demucs", default="demucs", help="Name of the pretrained demucs.")
    model_config.add_argument("--wav2vec", default="facebook/wav2vec2-base-960h",
                              help="Name of the pretrained wav2vec.")
    model_config.add_argument("--tokenizer", default="facebook/wav2vec2-base-960h",
                              help="Name of the pretrained tokenizer.")
    model_config.add_argument("--freeze-demucs", action="store_true", help="Does not compute gradient on demucs model")
    model_config.add_argument("--freeze-extractor", action="store_true",
                              help="Freeze feature extractor layers from wav2vec.")

    model_io = parser.add_argument_group('model IO arguments', 'parameters related with load/save model')
    model_io.add_argument("--load-model", type=pathlib.Path, help="Loads the specified model.")
    model_io.add_argument("--model-folder", type=pathlib.Path,
                          help="Folder where the model will be saved per epoch and when signaled with SIGUSR.")
    model_io.add_argument("--save-on-epoch", type=pathlib.Path, help="If specified, saves the model on every epoch.")

    args = parser.parse_args()

    batch_size = args.batch

    # User input validation and transformation
    DALI_DATA_PATH = (args.data_path / "dali").resolve(strict=True)
    DALI_AUDIO_PATH = (args.data_path / "audio").resolve(strict=True)
    TEXT_ARPA = (args.lm or (args.data_path / "text.arpa")).resolve(strict=True)
    CHECKPOINT_FOLDER = (args.model_folder or (args.data_path / "checkpoint")).resolve()
    CHECKPOINT_FOLDER.mkdir(parents=True, exist_ok=True)

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
    dali_data = dali_code.get_the_DALI_dataset(str(DALI_DATA_PATH), gt_file=gt_file, skip=blacklist, keep=[])

    if args.ncc:
        dataset = DaliDataset(dali_data, DALI_AUDIO_PATH, length=audio_length, stride=stride, ncc=(args.ncc, None),
                              workers=args.workers)
        train_len = int(len(dataset) * args.train_split)
        val_len = int((len(dataset) * (1 - args.train_split)) / 2)
        test_len = len(dataset) - train_len - val_len
        train_dataset, validation_dataset, test_dataset = \
            random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))
        assert len(train_dataset) > 0 and len(validation_dataset) > 0, "No data selected with these parameters"
        print(f"Train DaliDataset: {len(train_dataset)} chunks")
        print(f"Validation DaliDataset: {len(validation_dataset)} chunks")
        print(f"Test DaliDataset: {len(test_dataset)} chunks")
    else:

        print("Preparing Datasets...")
        train_dataset = DaliDataset(dali_data, DALI_AUDIO_PATH, length=audio_length, stride=stride, ncc=(.8, .925),
                                    workers=args.workers)
        print(f"Train DaliDataset: {len(train_dataset)} chunks")
        validation_dataset = DaliDataset(dali_data, DALI_AUDIO_PATH, length=audio_length, stride=stride,
                                         ncc=(.925, .94), workers=args.workers)
        print(f"Validation DaliDataset: {len(validation_dataset)} chunks")
        test_dataset = DaliDataset(dali_data, DALI_AUDIO_PATH, length=audio_length, stride=stride, ncc=(.94, None),
                                   workers=args.workers)
        print(f"Test DaliDataset: {len(test_dataset)} chunks")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Beam decoder
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    vocab = [x[1].replace("|", " ") if x[1] not in tokenizer.all_special_tokens else "_" for x in sort_vocab]
    vocabulary = vocab
    beam_decoder = ctcdecode.BeamSearchDecoder(
        vocabulary,
        num_workers=args.workers or 4,
        beam_width=128,
        cutoff_prob=np.log(0.000001),
        cutoff_top_n=40
    )

    # KenLM
    alpha = 2.5  # LM Weight
    beta = 0.0  # LM Usage Reward
    word_lm_scorer = ctcdecode.WordKenLMScorer(str(TEXT_ARPA), alpha, beta)
    beam_lm_decoder = ctcdecode.BeamSearchDecoder(
        vocabulary,
        num_workers=args.workers or 4,
        beam_width=128,
        scorers=[word_lm_scorer],
        cutoff_prob=np.log(0.000001),
        cutoff_top_n=40)

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

    config = {
        'audio_length': args.audio_length,
        'stride': args.stride,
        'epochs': args.epochs,
        'batch_size': args.batch,
        'learning_rate': args.lr,
        'weight_decay': args.wd,
        'optimizer': args.optimizer,
        # 'dropout': args.dropout,
        'workers': args.workers,
        'freeze_demucs': args.freeze_demucs,
        'freeze_extractor': args.freeze_extractor,
        'train_len': len(train_dataset),
        'validation_len': len(validation_dataset),
        'test_len': len(test_dataset)
    }

    # WANDB configuration
    if 'WANDB_KEY' in os.environ:
        wandb.login(key=os.environ['WANDB_KEY'])
        os.environ["WANDB_SILENT"] = "true"
        os.environ["WANDB_MODE"] = "offline"
    wandb.init(config=config)
    config = wandb.config

    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu)

    # Load the model
    model = DemucsWav2Vec(demucs=args.demucs, wav2vec=args.wav2vec)

    # Setup optimizer and LR scheduler
    # Define the optimizer
    if args.optimizer == 'sgd':
        optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise RuntimeError("No Optimizer specified")

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    criterion = torch.nn.CTCLoss()

    if args.freeze_demucs:
        for param in model.demucs.parameters():
            param.requires_grad = False

    if args.freeze_extractor:
        model.wav2vec.freeze_feature_extractor()
        # model.wav2vec.wav2vec2.feature_projection.requires_grad_(False)

    wandb.watch(model)

    if args.load_model:
        print("Loading model from ", args.load_model)
        load_model(args.load_model, model, optimizer)

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader)

    def handler(signum, frame):
        save_model(model, optimizer, folder=CHECKPOINT_FOLDER, train_loss=None, val_loss=None, accelerator=accelerator)

    signal.signal(signal.SIGUSR1, handler)

    accelerator.print("Start Training with device", str(accelerator.device))
    losses = {'train': [], 'valid': []}
    for epoch in range(config.epochs):
        start_time = time.time()
        model.train()
        train_losses = []
        for idx, batch in enumerate(train_loader):
            waveforms, lyrics = batch
            logits, voice = model(waveforms)
            batch_size, input_lengths, classes = logits.size()
            target_lengths = lyrics.count_nonzero(dim=1)
            log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            loss = criterion(log_prob, lyrics,
                             input_lengths=torch.full(size=(batch_size,), fill_value=input_lengths, dtype=torch.short),
                             target_lengths=target_lengths)

            train_losses.append(loss.item())

            accelerator.backward(loss)

            optimizer.step()
            # if scheduler:
            #     scheduler.step(loss)

            wandb.log({"batch_train_loss": loss.item()})
            optimizer.zero_grad()

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted = tokenizer.decode(predicted_ids[0])
        wandb.log({"input": wandb.Audio(waveforms[0].mean(0).cpu().numpy(),
                                        sample_rate=model.demucs.samplerate),
                   "voice": wandb.Audio(voice[0].cpu().numpy(), sample_rate=model.sr_wav2vec),
                   "predictions": wandb.Html(f"""<table style="width:100%">
                   <tr><th>Epoch</th> <th>Batch ID</th> <th>Lyric</th> <th>Predicted</th> </tr>
                   <tr><td>{epoch}</td>
                   <td>{idx}</td>
                   <td>{tokenizer.decode(lyrics[0])}</td>
                   <td>{predicted}</td></tr>
                   </table>"""),
                   "epoch": epoch})

        val_losses = []
        model.eval()
        for waveforms, lyrics in val_loader:
            with torch.no_grad():
                logits, voice = model(waveforms)
            batch_size, input_lengths, classes = logits.size()
            target_lengths = lyrics.count_nonzero(dim=1)
            log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
            log_prob = accelerator.gather(log_prob)
            lyrics = accelerator.gather(lyrics)
            loss = criterion(log_prob, lyrics,
                             input_lengths=torch.full(size=(batch_size,), fill_value=input_lengths,
                                                      dtype=torch.short),
                             target_lengths=target_lengths)
            val_losses.append(loss.item())

        # Loss average
        average_train_loss = np.mean(train_losses)
        average_valid_loss = np.mean(val_losses)
        losses['train'].append(average_train_loss)
        losses['valid'].append(average_valid_loss)

        wandb.log({"train_loss": average_train_loss, "valid_loss": average_valid_loss})

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        accelerator.print(f"EPOCH: {epoch + 1},  | time in {int(mins)} minutes, {secs} seconds")
        # print progress
        accelerator.print(f'=> train loss: {average_train_loss:0.3f}  => valid loss: {average_valid_loss:0.3f}')

        if args.model_folder and args.save_on_epoch:
            save_model(model, optimizer, CHECKPOINT_FOLDER, train_loss=np.mean(losses["train"]),
                       val_loss=np.mean(losses["valid"]), epoch=epoch, accelerator=accelerator)

    accelerator.print("Training finished")

    save_model(model, optimizer, CHECKPOINT_FOLDER, train_loss=np.mean(losses["train"]),
               val_loss=np.mean(losses["valid"]), accelerator=accelerator)


    if test_loader:
        accelerator.print("Testing the model...")
        model.eval()
        test_loss = []
        wers = []
        table = wandb.Table(
            columns=["lyric", "predicted", "wer", "beam predicted", "beam wer", "beam LM predicted", "beam LM wer"])
        with torch.no_grad():
            for waveforms, lyrics in test_loader:
                with torch.no_grad():
                    logits, voice = model(waveforms)
                batch_size, input_lengths, classes = logits.size()
                log_prob = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
                log_prob = accelerator.gather(log_prob)
                lyrics = accelerator.gather(lyrics)
                target_lengths = lyrics.count_nonzero(dim=1)
                loss = criterion(log_prob, lyrics,
                                 input_lengths=torch.full(size=(batch_size,), fill_value=input_lengths,
                                                          dtype=torch.short),
                                 target_lengths=target_lengths)
                test_loss.append(loss.item())

                # WER calculation
                ground_truth = tokenizer.batch_decode(lyrics)
                predicted = tokenizer.batch_decode(torch.argmax(logits, dim=-1).detach().cpu())
                beam_predicted = beam_decoder.decode_batch(F.log_softmax(logits, dim=-1).detach().cpu())
                kenlm_predicted = beam_lm_decoder.decode_batch(F.log_softmax(logits, dim=-1).detach().cpu())
                for i, lyric in enumerate(lyrics):
                    wer = jiwer.wer(ground_truth[i], predicted[i])
                    beam_wer = jiwer.wer(ground_truth[i], beam_predicted[i])
                    beam_lm_wer = jiwer.wer(ground_truth[i], kenlm_predicted[i])
                    wers.append((wer, beam_wer, beam_lm_wer))
                    table.add_data(tokenizer.decode(lyric), predicted[i], wer * 100, beam_predicted[i], beam_wer * 100,
                                   kenlm_predicted[i], beam_lm_wer * 100)

        test_loss = np.mean(test_loss)
        test_wer = np.mean(wers, axis=0)
        accelerator.print(f'Test set: Average loss: {test_loss:.4f}, Wer: {test_wer[0] * 100:.2f}%, '
                          f'Beam Wer: {test_wer[1] * 100:.2f}%, Beam LM Wer: {test_wer[2] * 100:.2f}%')
        wandb.log({"predictions": table})
        wandb.run.summary['wer'] = test_wer[0] * 100
        wandb.run.summary['beam wer'] = test_wer[1] * 100
        wandb.run.summary['beam lm wer'] = test_wer[2] * 100
        wandb.run.summary['test_loss'] = test_loss


if __name__ == "__main__":
    main()

