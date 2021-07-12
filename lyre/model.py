import julius
import torch
import torch.nn as nn

from demucs.pretrained import load_pretrained
from demucs.utils import center_trim, tensor_chunk
from transformers import Wav2Vec2ForCTC


class DemucsWav2Vec(nn.Module):
    def __init__(self, demucs="demucs", wav2vec="facebook/wav2vec2-base-960h", wav2vec_kwargs=None, sr_wav2vec=16000):
        super().__init__()
        self.demucs = load_pretrained(demucs)
        self.sr_wav2vec = sr_wav2vec
        self.resample = julius.resample.ResampleFrac(self.demucs.samplerate, self.sr_wav2vec)
        self.wav2vec = Wav2Vec2ForCTC.from_pretrained(wav2vec, **(wav2vec_kwargs or {}))

    def forward(self, mix, labels=None):
        batch, channels, length = mix.shape
        mix_chunk = tensor_chunk(mix)
        valid_length = self.demucs.valid_length(length)

        padded_mix = mix_chunk.padded(valid_length)
        sources = self.demucs(padded_mix)
        sources = center_trim(sources, length)

        # Extract voice and squeeze:
        voice = sources[:, 3, :, :]

        # transform from stereo to mono:
        output_voice_mono = voice.mean(dim=1)

        # change sample rate:
        try:
            output_voice_mono_sr = self.resample(output_voice_mono)
        except:
            print(output_voice_mono.size())
            raise

        # Wav2Vec processor function:
        input_values = (output_voice_mono_sr - output_voice_mono_sr.mean(1, keepdim=True)) / torch.sqrt(
            output_voice_mono_sr.var(1, keepdim=True) + 1e-5)

        # Wav2Vec:
        wav2vec_output = self.wav2vec(input_values, labels=labels)

        return wav2vec_output, output_voice_mono_sr
