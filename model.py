import torch
import torch.nn as nn
import torchaudio

#Import models:

from demucs.pretrained import load_pretrained
from transformers import AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

#Construct the model

class DemucsWav2Vec(nn.Module):
    def __init__(self):
        super().__init__()
        self.demucs = load_pretrained("demucs")
        self.wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.changesamplerate = torchaudio.transforms.Resample(44100, 16000)

    def forward(self, input_tensor):
        # Demucs:
        out_demucs = self.demucs(input_tensor)

        # Extract voice and squeeze:
        output_voice = out_demucs[:, 3, :, :]

        # transform from stereo to mono:
        output_voice_mono = torch.mean(output_voice, dim=1)

        # change sample rate:

        output_voice_mono_sr = self.changesamplerate(output_voice_mono)

        # Wav2Vec processor function:

        input_values = (output_voice_mono_sr - output_voice_mono_sr.mean(1)) / torch.sqrt(
            output_voice_mono_sr.var(1) + 1e-5)

        # Wav2Vec:
        logits = self.wav2vec(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.wav2vec_processor.decode(predicted_ids[0])

        return transcription

Mymodel = DemucsWav2Vec()