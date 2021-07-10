import math
import multiprocessing
import pathlib
import re
import typing as t
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

import DALI as dali_code
import julius
import torch
import torchaudio
from torch.utils.data import Dataset

from .utils import time2sample

"""
entry.info --> {'id': 'a_dali_unique_id',
                'artist': 'An Artist',
                'title': 'A song title',
                'dataset_version': 1.0,     **# dali_data version**
                'ground-truth': False,
                'scores': {'NCC': 0.8098520072498807,
                           'manual': 0.0},  **# Not ready yet**
                'audio': {'url': 'a youtube url',
                          'path': 'None',
                          **# To you to modify it to point to your local audio file**
                          'working': True},
                'metadata': {'album': 'An album title',
                             'release_date': 'A year',
                             'cover': 'link to a image with the cover',
                             'genres': ['genre_0', ... , 'genre_n'],
                             # The n of genre depends on the song
                             'language': 'a language'}}

entry.annotations --> {'annot': {'the annotations themselves'},
                       'type': 'horizontal' or 'vertical',
                       'annot_param': {'fr': float(frame rate used in the annotation process),
                                      'offset': float(offset value)}}

{'text': 'wo', # the annotation itself.
                 'time': [12.534, 12.659], # the begining and end of the  segment in seconds.
                 'freq': [466.1637615180899, 466.1637615180899], # The range of frequency the text information is covering. At the lowest level, syllables, it corresponds to the vocal note.
                 'index': 0} # link with the upper level. For example, index 0 at the 'words' level means that that particular word below to first line ([0]). The paragraphs level has no index key.
"""


@dataclass
class Chunk:
    song_id: str
    init_sample: int
    end_sample: int
    audio_start: int
    audio_end: int
    lyrics: str


def _process_file(entry, samplerate, length, stride):
    chunk_map = []
    file = entry.info["audio"]["path"]
    sample_rate = entry.info["audio"]["metadata"].sample_rate
    track_length = int(samplerate / sample_rate * entry.info["audio"]["metadata"].num_frames)
    if length is None or track_length < length:
        chunks = 1
    else:
        chunks = int(math.ceil((track_length - length) / stride) + 1)

    notes = iter(entry.annotations["annot"]["notes"])
    lyric = " ".join([p["text"] for p in entry.annotations["annot"]["paragraphs"]])
    current_note = None
    for chunk in range(chunks):
        if length is None:
            chunk_start = 0
            chunk_end = track_length - 1
        else:
            chunk_start = chunk * length
            chunk_end = (chunk + 1) * length - 1
        audio_start = chunk_start

        if chunk == (chunks - 1):
            # Final chunk
            audio_end = track_length - 1
        else:
            audio_end = chunk_end

        text_notes = []
        skip_chunk = False
        while True:
            if current_note is None:
                try:
                    current_note = next(notes)
                    time_note_start, time_note_end = current_note["time"]
                    if time_note_start < 0 or time_note_end < 0:
                        # print(f"Negative notes in {entry.info['id']}")
                        return []
                    note_start = time2sample(time_note_start, sample_rate=samplerate)
                    note_end = time2sample(time_note_end, sample_rate=samplerate)
                except StopIteration:
                    break
            if note_start > chunk_end:
                break
            elif note_start < audio_start:

                if note_end < chunk_end:
                    audio_start = note_end + 1
                else:
                    # note is larger than the chunk
                    skip_chunk = True
                    break
                current_note = None
                continue
            elif audio_start <= note_start and note_end <= chunk_end:
                text_notes.append(re.escape(current_note["text"].replace("~", "")))
                current_note = None
            elif note_end > chunk_end:
                audio_end = note_start - 1
                break

        if not skip_chunk:
            try:
                match = re.search("\s?".join(text_notes), lyric)
                if text_notes and match:
                    text = match.group(0)
                else:
                    text = ""
            except:
                print(text_notes)
            chunk = Chunk(
                entry.info['id'],
                chunk_start,
                chunk_end,
                audio_start,
                audio_end,
                text.upper(),
            )
            if not 0 <= chunk_start <= audio_start < audio_end <= chunk_end or not text:
                # print("Invalid Chunk: ", chunk)
                pass
            else:
                chunk_map.append(chunk)
    return chunk_map


DEFAULT_SAMPLE_RATE = 44100


class DaliDataset(Dataset):

    def __init__(self, dali_data, dali_audio_path, gt_file=None, blacklist=None, length=None, stride=None,
                 normalize=True, samplerate=DEFAULT_SAMPLE_RATE,
                 ncc: t.Tuple[t.Optional[float], t.Optional[float]] = None,
                 workers=0):
        """

        Args:
            dali_data: folder containing dali data or the loaded dali dataset
            dali_audio_path: path to the dali audio files
            gt_file: Dali's Ground truth file
            length: chunk's samples length
            stride: stride applied on chunks
            normalize: normalize chunk audio
            samplerate: output sample rate
            ncc: values used for filtering dali dataset. NCC must be between min and max. Value must be between 0 and 1
        """
        if isinstance(dali_data, dict):
            self.dali_data = dali_data
        else:
            self.dali_data = None
            if isinstance(dali_data, str):
                dali_data = pathlib.Path(dali_data)
            self.dali_data_path = dali_data.resolve()
        self.dali_data_subset_ident = []

        if isinstance(dali_audio_path, str):
            dali_audio_path = pathlib.Path(dali_audio_path)
        self.dali_audio_path = dali_audio_path

        if isinstance(gt_file, str):
            gt_file = pathlib.Path(gt_file)
        self.gt_file = gt_file

        self.blacklist = blacklist or []
        self.length = length
        self.stride = stride or self.length
        self.normalize = normalize
        self.samplerate = samplerate
        self.chunk_map: t.List[Chunk] = []
        self.workers = workers
        if ncc is not None:
            self.min_ncc = ncc[0] or 0
            self.max_ncc = ncc[1] or 1
        else:
            self.min_ncc = 0
            self.max_ncc = 1
        self._load_data()

    def _load_data(self):
        if not self.dali_data:
            print("Loading DALI data from ", self.dali_data_path)
            self.dali_data = dali_code.get_the_DALI_dataset(self.dali_data_path, gt_file=self.gt_file or '', skip=self.blacklist)

        # print("Generating dataset information")
        files = list(self.dali_audio_path.glob("*.mp3"))

        for file in files:
            iden = file.stem
            if iden in self.dali_data and self.dali_data[iden].info["metadata"]["language"] == "english" and \
                self.min_ncc <= self.dali_data[iden].info["scores"]["NCC"] < self.max_ncc:
                self.dali_data[iden].info["audio"]["path"] = file.absolute()
                self.dali_data[iden].info["audio"]["metadata"] = torchaudio.info(file.absolute())
                self.dali_data_subset_ident.append(iden)

        # print("Generate Chunks")

        if self.workers == 0 or self.workers is None:
            workers = max(multiprocessing.cpu_count() - 1, 1)
        else:
            workers = self.workers

        if workers > 1:
            with Pool(workers) as p:
                result = p.map(
                    partial(_process_file, samplerate=self.samplerate, length=self.length, stride=self.stride),
                    [self.dali_data[i] for i in self.dali_data_subset_ident])

            for chunks in result:
                if chunks:
                    self.chunk_map.extend(chunks)
        else:
            for i in self.dali_data_subset_ident:
                chunks = _process_file(self.dali_data[i], samplerate=self.samplerate, length=self.length,
                                       stride=self.stride)
                if chunks:
                    self.chunk_map.extend(chunks)

    def __len__(self):
        return len(self.chunk_map)
        # return 1

    def __getitem__(self, idx):
        chunk_meta = self.chunk_map[idx]

        entry = self.dali_data[chunk_meta.song_id]
        sample_rate = entry.info["audio"]["metadata"].sample_rate
        try:
            waveform, sample_rate = torchaudio.load(
                entry.info["audio"]["path"],
                frame_offset=int(sample_rate / self.samplerate * chunk_meta.audio_start),
                num_frames=int(sample_rate / self.samplerate * (chunk_meta.audio_end - chunk_meta.audio_start + 1)),
            )
        except:
            print(chunk_meta)
            raise
        channels = waveform.size()[0]

        waveform = julius.resample_frac(waveform, sample_rate, self.samplerate)

        start_silence = chunk_meta.audio_start - chunk_meta.init_sample
        end_silence = chunk_meta.end_sample - chunk_meta.audio_end
        if self.length is not None:
            end_silence = end_silence + (self.length - (start_silence + waveform.size()[1] + end_silence))
        try:
            waveform = torch.cat((torch.zeros(channels, start_silence), waveform, torch.zeros(channels, end_silence)),
                                 1)
        except:
            print(chunk_meta)
            raise
        if channels == 1:
            waveform = torch.stack((waveform, waveform)).squeeze()

        if self.normalize:
            waveform = waveform / max(1.01 * waveform.abs().max(), 1.0)

        return (waveform, chunk_meta.lyrics)
