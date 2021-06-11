import DALI as dali_code
import random
import pathlib
from dataclasses import dataclass
import typing as t
import re
import math
from tqdm import tqdm

from torch.utils.data import Dataset
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

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


class DaliDataset(Dataset):
    def __init__(self, root_dir, length=None, stride=None, normalize=True, samplerate=44100, channels=2):
        """[summary]

        Args:
            root_dir ([type]): [description]
            metadata ([type]): [description]
            length ([type], optional): [description]. Defaults to None.
            stride ([type], optional): [description]. Defaults to None.
            normalize (bool, optional): [description]. Defaults to True.
            samplerate (int, optional): [description]. Defaults to 44100.
            channels (int, optional): [description]. Defaults to 2.
        """
        if isinstance(root_dir, str):
            self.root_dir = pathlib.Path(root_dir).absolute()
        else:
            self.root_dir = root_dir.absolute()
        self.dali_data_path = self.root_dir / "dali"
        self.dali_audio_path = self.root_dir / "audio"
        self.length = length
        self.stride = stride or length
        self.normalize = normalize
        self.samplerate = samplerate
        self.channels = channels
        self.chunk_map: t.List[Chunk] = []
        self.dali_data = {}

        self._load_data()

    def _load_data(self):
        # ground_truth = dali_code.utilities.read_gzip(self.root_dir / "gt_v1.0_22_11_18.gz")
        print("Loading DALI data from", self.root_dir)
        dali_data = dali_code.get_the_DALI_dataset(self.dali_data_path, self.root_dir / "gt_v1.0_22_11_18.gz", skip=[],
                                                   keep=[])
        print("Generating dataset information")
        # dali_info = dali_code.get_info(DALI_DATA_PATH / 'info' / 'DALI_DATA_INFO.gz')
        files = list(self.dali_audio_path.glob("*.mp3"))
        with tqdm(total=len(files)) as pbar:
            for file in files:
                pbar.update(1)
                iden = file.stem
                entry = dali_data[iden]
                entry.info["audio"]["path"] = file.absolute()
                entry.info["audio"]["metadata"] = torchaudio.info(file.absolute())
                if entry.info["metadata"]["language"] == "english":
                    self.dali_data[iden] = entry
                    sample_rate = entry.info["audio"]["metadata"].sample_rate
                    track_length = int(self.samplerate / sample_rate * entry.info["audio"]["metadata"].num_frames)
                    if self.length is None or track_length < self.length:
                        chunks = 1
                    else:
                        chunks = int(math.ceil((track_length - self.length) / self.stride) + 1)

                    notes = iter(entry.annotations["annot"]["notes"])
                    lyric = " ".join([p["text"] for p in entry.annotations["annot"]["paragraphs"]])
                    current_note = None
                    for chunk in range(chunks):
                        chunk_start = chunk * self.length
                        audio_start = chunk_start
                        chunk_end = (chunk + 1) * self.length - 1
                        if chunk == (chunks - 1):
                            audio_end = track_length - 1
                        else:
                            audio_end = chunk_end
                        text_notes = []

                        while True:
                            if current_note is None:
                                try:
                                    current_note = next(notes)
                                    time_note_start, time_note_end = current_note["time"]
                                    note_start = time2sample(time_note_start, sample_rate=self.samplerate)
                                    note_end = time2sample(time_note_end, sample_rate=self.samplerate)
                                except StopIteration:
                                    break
                            if note_start > chunk_end:
                                break
                            elif note_start < audio_start:
                                audio_start = note_end + 1
                                current_note = None
                                continue
                            elif audio_start <= note_start and note_end <= chunk_end:
                                text_notes.append(re.escape(current_note["text"].replace("~", "")))
                                current_note = None
                            elif note_end > chunk_end:
                                audio_end = note_start - 1
                                break

                        try:
                            if text_notes and (match := re.search("\s?".join(text_notes), lyric)):
                                text = match.group(0)
                            else:
                                text = ""
                        except:
                            print(text_notes)

                        self.chunk_map.append(
                            Chunk(
                                iden,
                                chunk_start,
                                chunk_end,
                                audio_start,
                                audio_end,
                                text,
                            )
                        )

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        chunk_meta = self.chunk_map[idx]

        entry = self.dali_data[chunk_meta.song_id]
        sample_rate = entry.info["audio"]["metadata"].sample_rate
        waveform, sample_rate = torchaudio.load(
            entry.info["audio"]["path"],
            frame_offset=int(sample_rate / self.samplerate * chunk_meta.audio_start),
            num_frames=int(sample_rate / self.samplerate * (chunk_meta.audio_end - chunk_meta.audio_start + 1)),
        )
        channels = waveform.size()[0]
        waveform = T.Resample(sample_rate, self.samplerate)(waveform)

        start_silence = chunk_meta.audio_start - chunk_meta.init_sample
        end_silence = chunk_meta.end_sample - chunk_meta.audio_end
        end_silence = end_silence + (self.length - (start_silence + waveform.size()[1] + end_silence))
        waveform = torch.cat((torch.zeros(channels, start_silence), waveform, torch.zeros(channels, end_silence)), 1)

        return {"waveform": waveform, "lyrics": chunk_meta.lyrics}
