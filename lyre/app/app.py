import io
import math
import os
import pathlib
import time
import uuid
from multiprocessing import Pool

import ctcdecode
import kenlm
import numpy as np
import torch
import julius
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
from torch.nn import functional as F
from torch import nn
import soundfile as sf
from flask import Flask, render_template, request, send_from_directory, session, url_for
from transformers import AutoTokenizer
from werkzeug.utils import secure_filename, redirect

from lyre.model import DemucsWav2Vec

ROOT = pathlib.Path(__file__).parent.absolute()
app = Flask(__name__, template_folder=str(ROOT / "template"))

CHECKPOINT = "/home/joan/AIDL/aidl-lyrics-recognition/data/checkpoint/model_epoch_b6a4718e.pt"
TEXT_ARPA = "/home/joan/AIDL/aidl-lyrics-recognition/data/text.arpa"
CHUNK_LENGTH = 5  # in seconds
MAX_BATCH = 64
MAX_LENGTH = 60  # Max length in seconds of audio processed

MODEL: nn.Module = None
TOKENIZER = None
LM = None
DEVICE = os.environ.get("DEVICE", None)

if DEVICE:
    DEVICE = torch.device(DEVICE)
else:
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

@app.before_first_request
def _load_model():
    # First load into memory the variables that we will need to predict
    checkpoint = torch.load(CHECKPOINT, map_location=torch.device('cpu'))

    global MODEL, TOKENIZER, LM

    MODEL = DemucsWav2Vec().eval()
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    TOKENIZER = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

    kenlm_model = kenlm.Model(str(TEXT_ARPA))
    # Beam decoder
    # make alphabet
    vocab_list = list(TOKENIZER.get_vocab().keys())
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

    LM = BeamSearchDecoderCTC(alphabet, LanguageModel(kenlm_model, alpha=.5, beta=5.))


app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = {'.mp3', '.wav'}
app.config['UPLOAD_PATH'] = '/tmp/uploads'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['SECRET_KEY'] = app.secret_key


def validate_audio(stream):
    return True


@app.errorhandler(413)
def too_large(e):
    return "File is too large", 413


#
# @app.route('/')
# def index():
#     print("Index ", session.get('lyric'))
#     template = render_template('index.html', lyric=request.args.get('lyric', ''))
#     return template
import torchaudio as ta


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        start_time = time.time()
        uploaded_file = request.files['file']
        filename = secure_filename(uploaded_file.filename)
        if filename != '':
            print("Processing file ", filename)
            uploaded_file.save('/tmp/' + uploaded_file.filename)
            try:
                waveform, samplerate = ta.load('/tmp/' + uploaded_file.filename)
            except RuntimeError as e:
                app.logger.info(e)
                return render_template('index.html', message="Format not recognised"), 400

            waveform = julius.resample_frac(waveform, samplerate, MODEL.demucs.samplerate)
            waveform = waveform[:, :MAX_LENGTH * MODEL.demucs.samplerate]
            chunk_length = CHUNK_LENGTH * MODEL.demucs.samplerate
            channels, length = waveform.size()

            waveform = torch.cat([waveform, torch.zeros(channels, length % chunk_length)], dim=1)
            slices = waveform.chunk(int(length / chunk_length), dim=1)
            if len(slices) > 1:
                waveform = torch.stack(slices[:-1], dim=0)
            else:
                waveform = slices[0].unsqueeze(0)
            del slices

            process_file_time = time.time() - start_time
            start_time = time.time()
            chunks = waveform.size(0)
            list_log_probs = []
            for i in range(math.ceil(chunks / MAX_BATCH)):
                with torch.no_grad():
                    output, _ = MODEL(waveform[i * MAX_BATCH:(i + 1) * MAX_BATCH])
                list_log_probs.append(F.log_softmax(output.logits, dim=-1))
            list_all_probs = [log_probs[i] for log_probs in list_log_probs for i in range(log_probs.size()[0])]
            # del list_log_probs, waveform
            # remove last
            if len(list_all_probs) > 1:
                list_all_probs = list_all_probs[:-1]

            all_prob = torch.cat(list_all_probs).detach().cpu().numpy()
            # del list_all_probs

            forward_time = time.time() - start_time
            print(f"Inference Time: {int(forward_time / 60)}:{forward_time:.0f} sec")
            start_time = time.time()
            # log_probs = [x.squeeze(0).numpy() for x in torch.chunk(all_prob, all_prob.size(0), dim=0)]
            # WER calculation
            lyric = LM.decode(all_prob)
            lm_time = time.time() - start_time
            session["lyric"] = lyric
            session["process_file_time"] = process_file_time
            session["forward_time"] = forward_time
            session["lm_time"] = lm_time

        return redirect(url_for('index'))

    if session.get('lyric'):
        message = f"Pre-Processing File Time: {int(session.get('process_file_time', .0) / 60)}:{int(session.get('process_file_time', .0) % 60)} sec\n" \
                  f"Inference Time: {int(session.get('forward_time', .0) / 60)}:{int(session.get('forward_time', .0) % 60)} sec\n" \
                  f"Language Model File Time: {int(session.get('lm_time', .0) / 60)}:{int(session.get('lm_time', .0) % 60)} sec\n" \
                  f"Lyric: {session['lyric']}"
    else:
        message = ''
    return render_template('index.html', message=message)


@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)


if __name__ == '__main__':
    app.run(debug=True)
