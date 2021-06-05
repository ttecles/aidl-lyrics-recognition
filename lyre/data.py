import DALI as dali_code
import random
import pathlib
from dataclasses import dataclass
import typing as t
import re
import math

from torch.utils.data import Dataset
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
        self.root_dir = root_dir
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
        dali_data = dali_code.get_the_DALI_dataset(self.dali_data_path, self.root_dir / "gt_v1.0_22_11_18.gz", skip=[], keep=[])
        # dali_info = dali_code.get_info(DALI_DATA_PATH / 'info' / 'DALI_DATA_INFO.gz')

        for file in self.dali_audio_path.glob("*.mp3"):
            iden = file.stem
            entry = dali_data[iden]
            entry.info["audio"]["path"] = file.absolute()
            entry.info["audio"]["metadata"] = torchaudio.info(file.absolute())
            if entry.info["metadata"]["language"] == "english":
                self.dali_data[iden] = entry
                sample_rate = entry.info["audio"]["metadata"].sample_rate
                track_length = int(self.samplerate / sample_rate * entry.info["audio"]["metadata"].num_frames )
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
                    audio_end = chunk_end
                    text_notes = []

                    while True:
                        if current_note is None:
                            current_note = next(notes)
                            note_start, note_end = current_note["time"]
                            note_start = time2sample(note_start, sample_rate=self.samplerate)
                            note_end = time2sample(note_end, sample_rate=self.samplerate)
                        if note_start > chunk_end:
                            break
                        elif note_start < audio_start:
                            audio_start = note_end + 1
                            current_note = None
                            continue
                        elif audio_start <= note_start and note_end <= chunk_end:
                            text_notes.append(current_note["text"])
                        elif note_end > chunk_end:
                            audio_end = note_start - 1
                            break

                    if text_notes and (match := re.search("\s?".join(text_notes), lyric)):
                        text = match.group(0)
                    else:
                        text = ""

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

    # def __getitem__(self, idx):
    #     chunk_meta = self.chunk_map[idx]

    #     entry = self.dali_data[chunk_meta.song_id]
    #     waveform, sample_rate = torchaudio.load(
    #         entry.info["audio"]["path"],
    #         frame_offset=chunk_meta.init_sample,
    #         num_frames=self.chunk_size,
    #     )

    #     init_chunk_time = frame2time(chunk_meta.init_sample, sample_rate)
    #     end_chunk_time = frame2time(chunk_meta.end_sample, sample_rate)

    #     return {"waveform": waveform, "lyrics": None}
