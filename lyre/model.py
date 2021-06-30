import nltk as nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from demucs.pretrained import load_pretrained
from transformers import Wav2Vec2ForCTC


class DemucsWav2Vec(nn.Module):
    def __init__(self):
        super().__init__()

        self.demucs = load_pretrained("demucs")
        self.resample = torchaudio.transforms.Resample(44100, 16000)
        self.wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    def forward(self, input_tensor):
        # Demucs:
        out_demucs = self.demucs(input_tensor)

        # Extract voice and squeeze:
        output_voice = out_demucs[:, 3, :, :]

        # transform from stereo to mono:
        output_voice_mono = torch.mean(output_voice, dim=1)

        # change sample rate:
        output_voice_mono_sr = self.resample(output_voice_mono)

        # Wav2Vec processor function:
        input_values = (output_voice_mono_sr - output_voice_mono_sr.mean(1, keepdim=True)) / torch.sqrt(
            output_voice_mono_sr.var(1, keepdim=True) + 1e-5)

        # Wav2Vec:
        logits = self.wav2vec(input_values).logits
        log_prob = F.log_softmax(logits, dim=-1)

        return log_prob
