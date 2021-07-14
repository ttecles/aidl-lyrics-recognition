import argparse
import os
import pathlib
import signal
import sys
import tempfile
import time
import typing as t
import uuid
from multiprocessing import Pool

import DALI as dali_code
import ctcdecode
import jiwer
import kenlm
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from dotenv import load_dotenv
from pyctcdecode import BeamSearchDecoderCTC, LanguageModel, Alphabet
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer

from lyre.data import DaliDataset, DEFAULT_SAMPLE_RATE
from lyre.model import DemucsWav2Vec

load_dotenv()


def save_model(model, optimizer: optim.Optimizer, folder: pathlib.Path, train_loss=None, val_loss=None, epoch=None,
               accelerator: Accelerator = None, name=None):
    name = name or f"model_{int(time.time())}.pt"

    try:
        folder.mkdir(parents=True, exist_ok=True)
    except:
        print(f"failed creating {folder}. Using temp dir")
        folder = pathlib.Path(tempfile.gettempdir())

    filename = (folder / name).resolve()

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


def load_model(file: t.Union[str, pathlib.Path], model=None, optimizer=None, map_location=None):
    # First load into memory the variables that we will need to predict
    checkpoint = torch.load(file, map_location=map_location)

    model = model or DemucsWav2Vec()
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_stage_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def parse_args():
    parser = argparse.ArgumentParser(prog="train")
    data = parser.add_argument_group('data arguments', 'data input information')
    data.add_argument("--data-path", type=pathlib.Path, help="Path holding all the data.", default='./data')
    data.add_argument("--dali-gt-file", type=pathlib.Path, help="DALI ground truth file.")
    data.add_argument("--blacklist-file", type=pathlib.Path,
                      help="All the identifiers listed in the file will be skipped from being loaded.")
    data.add_argument("--lm", type=pathlib.Path,
                      help="Trained Language Model file. if not specified it will try to find it in ./data/text.arpa")

    train_config = parser.add_argument_group('train config arguments', 'configuration of the training.')
    # Used for `distribution.launch`
    train_config.add_argument("--local_rank", type=int, default=os.environ.get('LOCAL_RANK', -1), metavar="N",
                              help="Local process rank.")
    train_config.add_argument("--log_all", action="store_true",
                              help="Flag to log in all processes, otherwise only in rank0.", )
    train_config.add_argument("--ncc", type=float, default=0,
                              help="Train only with files with NCC score bigger than NCC.")
    train_config.add_argument("--train-split", type=float, default=0.7,
                              help="Train proportion. Requires --ncc to be specified.")
    train_config.add_argument("--audio-length", type=int, default=5,
                              help="Audio length in seconds to pass to the model.")
    train_config.add_argument("--stride", type=int, default=None, help="Stride used for spliting the audio songs.")
    train_config.add_argument("--epochs", type=int, default=15, help="Number of epochs during training.")
    train_config.add_argument("--batch", type=int, default=8, help="Batch size.")
    train_config.add_argument("--drop-last", action="store_true",
                              help="set to True to drop the last incomplete batch, if the dataset size is not divisible"
                                   " by the batch size. If False and the size of dataset is not divisible by the batch "
                                   "size, then the last batch will be smaller.")
    train_config.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Type of optimizer.")
    train_config.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    train_config.add_argument("--wd", type=float, default=1e-4, help="Optimizer weight decay.")
    train_config.add_argument("--no-train", action="store_true", help="Executes the test, no training will be applied.")

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
    model_io.add_argument("--save-on-epoch", action="store_true", help="If specified, saves the model on every epoch.")

    return parser.parse_args()


def train(args):
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
                                    workers=args.workers) if not args.no_train else None
        print(f"Train DaliDataset: {len(train_dataset)} chunks") if not args.no_train else None
        validation_dataset = DaliDataset(dali_data, DALI_AUDIO_PATH, length=audio_length, stride=stride,
                                         ncc=(.925, .94), workers=args.workers) if not args.no_train else None
        print(f"Validation DaliDataset: {len(validation_dataset)} chunks") if not args.no_train else None
        test_dataset = DaliDataset(dali_data, DALI_AUDIO_PATH, length=audio_length, stride=stride, ncc=(.94, None),
                                   workers=args.workers)
        print(f"Test DaliDataset: {len(test_dataset)} chunks")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    kenlm_model = kenlm.Model(str(TEXT_ARPA))
    # Beam decoder
    # make alphabet
    vocab_list = list(tokenizer.get_vocab().keys())
    # convert ctc blank character representation
    vocab_list[0] = ""
    # replace special characters
    vocab_list[1] = "⁇"
    vocab_list[2] = "⁇"
    vocab_list[3] = "⁇"
    # convert space character representation
    vocab_list[4] = " "
    # specify ctc blank char index, since conventionally it is the last entry of the logit matrix
    alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=0)

    beam_decoder = BeamSearchDecoderCTC(alphabet)
    beam_lm_decoder = BeamSearchDecoderCTC(alphabet, LanguageModel(kenlm_model, alpha=.5, beta=5.))

    # build the decoder and decode the logits

    def collate(batch: list):
        # tokenizer.batch_decode(encoded)
        waveforms, lyrics = zip(*batch)
        lyrics_ids = tokenizer(lyrics, return_tensors='pt', padding='longest')['input_ids']
        return torch.stack(waveforms), lyrics_ids

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                              num_workers=args.workers, drop_last=args.drop_last) if not args.no_train else None
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
                            num_workers=args.workers, drop_last=args.drop_last) if not args.no_train else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
                             num_workers=args.workers, drop_last=args.drop_last)

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
        'train_len': len(train_dataset) if not args.no_train else None,
        'validation_len': len(validation_dataset) if not args.no_train else None,
        'test_len': len(test_dataset)
    }

    run = setup_run(args, config)
    # Check to see if local_rank is 0
    is_master = args.local_rank == 0
    do_log = run is not None

    handler = DistributedDataParallelKwargs()
    handler.find_unused_parameters = True
    accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, kwargs_handlers=[handler])

    # Load the model
    model = DemucsWav2Vec(demucs=args.demucs, wav2vec=args.wav2vec,
                          wav2vec_kwargs=dict(gradient_checkpointing=False,
                                              ctc_loss_reduction="mean",
                                              pad_token_id=tokenizer.pad_token_id))

    demucs_samplerate = model.demucs.samplerate
    wav2vec_samplerate = model.sr_wav2vec
    # Setup optimizer and LR scheduler
    # Define the optimizer
    if args.optimizer == 'sgd':
        optimizer: optim.Optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise RuntimeError("No Optimizer specified")

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    if args.freeze_demucs:
        for param in model.demucs.parameters():
            param.requires_grad = False

    if args.freeze_extractor:
        model.wav2vec.freeze_feature_extractor()
        # model.wav2vec.wav2vec2.feature_projection.requires_grad_(False)

    if is_master:
        wandb.watch(model)

    if args.load_model:
        print("Loading model from ", args.load_model)
        load_model(args.load_model, model, optimizer, map_location=torch.device('cpu'))

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader)

    def handler(signum, frame):
        save_model(model, optimizer, folder=CHECKPOINT_FOLDER, train_loss=None, val_loss=None, accelerator=accelerator)

    signal.signal(signal.SIGUSR1, handler)

    if not args.no_train:
        accelerator.print("Start Training with device", str(accelerator.device))
        losses = {'train': [], 'valid': []}
        model_dump_name = f"model_epoch_{str(uuid.uuid4()).split('-')[0]}.pt"
        for epoch in range(args.epochs):
            start_time = time.time()
            model.train()
            train_losses = []
            for idx, batch in enumerate(train_loader):
                waveforms, lyrics = batch
                lyrics[lyrics == 0] = -100
                output, voice = model(waveforms, labels=lyrics)
                lyrics[lyrics == -100] = 0
                train_losses.append(output.loss.item())

                accelerator.backward(output.loss)

                optimizer.step()
                # if scheduler:
                #     scheduler.step(loss)

                if do_log:
                    run.log({"batch_train_loss": train_losses[-1]})
                optimizer.zero_grad()

            predicted_ids = torch.argmax(output.logits, dim=-1)
            predicted = tokenizer.decode(predicted_ids[0])
            if do_log:
                run.log({"input": wandb.Audio(waveforms[0].mean(0).cpu().numpy(),
                                              sample_rate=demucs_samplerate),
                         "voice": wandb.Audio(voice[0].cpu().numpy(), sample_rate=wav2vec_samplerate),
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
                    lyrics[lyrics == 0] = -100
                    output, voice = model(waveforms, labels=lyrics)
                val_losses.append(accelerator.gather(output.loss.unsqueeze(0)).mean().item())

            # Loss average
            average_train_loss = np.mean(train_losses)
            average_valid_loss = np.mean(val_losses)
            losses['train'].append(average_train_loss)
            losses['valid'].append(average_valid_loss)

            if do_log:
                run.log({"train_loss": average_train_loss, "valid_loss": average_valid_loss})

            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            accelerator.print(f"EPOCH: {epoch + 1},  | time in {int(mins)} minutes, {secs} seconds")
            # print progress
            accelerator.print(f'=> train loss: {average_train_loss:0.3f}  => valid loss: {average_valid_loss:0.3f}')

            if args.model_folder and args.save_on_epoch:
                save_model(model, optimizer, CHECKPOINT_FOLDER, train_loss=np.mean(losses["train"]),
                           val_loss=np.mean(losses["valid"]), epoch=epoch, accelerator=accelerator,
                           name=model_dump_name)

        accelerator.print("Training finished")

        if args.model_folder and not args.save_on_epoch:
            save_model(model, optimizer, CHECKPOINT_FOLDER, train_loss=np.mean(losses["train"]),
                       val_loss=np.mean(losses["valid"]), accelerator=accelerator)

    if test_loader:
        accelerator.print("Testing the model...")
        model.eval()
        test_loss = []
        wers = []
        table = wandb.Table(
            columns=["lyric", "predicted", "wer", "beam predicted", "beam wer", "beam LM predicted", "beam LM wer"])
        with Pool(args.workers) as pool:
            with torch.no_grad():
                for waveforms, lyrics in tqdm(test_loader, unit="batch"):
                    with torch.no_grad():
                        lyrics[lyrics == 0] = -100
                        output, voice = model(waveforms, labels=lyrics)
                        lyrics[lyrics == -100] = 0
                    logits = accelerator.gather(output.logits)
                    lyrics = accelerator.gather(lyrics)
                    test_loss.append(accelerator.gather(output.loss.unsqueeze(0)).mean().item())
                    log_probs = F.log_softmax(logits, dim=-1).detach().cpu()
                    log_probs = [x.squeeze(0).numpy() for x in torch.chunk(log_probs, log_probs.size(0), dim=0)]
                    # WER calculation
                    ground_truth = tokenizer.batch_decode(lyrics)
                    predicted = tokenizer.batch_decode(torch.argmax(logits, dim=-1).detach().cpu())
                    beam_predicted = beam_decoder.decode_batch(pool, log_probs)
                    kenlm_predicted = beam_lm_decoder.decode_batch(pool, log_probs)
                    for i, lyric in enumerate(lyrics):
                        wer = jiwer.wer(ground_truth[i], predicted[i])
                        beam_wer = jiwer.wer(ground_truth[i], beam_predicted[i])
                        beam_lm_wer = jiwer.wer(ground_truth[i], kenlm_predicted[i])
                        wers.append((wer, beam_wer, beam_lm_wer))
                        table.add_data(tokenizer.decode(lyric), predicted[i], wer * 100, beam_predicted[i],
                                       beam_wer * 100,
                                       kenlm_predicted[i], beam_lm_wer * 100)

        test_loss = np.mean(test_loss)
        test_wer = np.mean(wers, axis=0)
        accelerator.print(f'Test set: Average loss: {test_loss:.4f}, Wer: {test_wer[0] * 100:.2f}%, '
                          f'Beam Wer: {test_wer[1] * 100:.2f}%, Beam LM Wer: {test_wer[2] * 100:.2f}%')
        if do_log:
            run.log({"predictions": table})
            run.summary['wer'] = test_wer[0] * 100
            run.summary['beam wer'] = test_wer[1] * 100
            run.summary['beam lm wer'] = test_wer[2] * 100
            run.summary['test_loss'] = test_loss

        wandb.finish()


def setup_run(args, config):
    run = None
    if args.log_all:
        run = wandb.init(group="DDP", config=config)
    elif args.local_rank == 0:
        run = wandb.init(config=config)
    elif args.local_rank == -1:
        if 'WANDB_KEY' in os.environ:
            wandb.login(key=os.environ['WANDB_KEY'])
            run = wandb.init(config=config)
    return run


if __name__ == "__main__":
    args = parse_args()

    train(args)
