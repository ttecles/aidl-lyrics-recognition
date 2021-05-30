import DALI as dali_code
import pathlib
from dataclasses import dataclass
import typing as t

from torch.utils.data import Dataset
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from utils import frame2time
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
"""

ROOT = pathlib.Path(__name__).parent.absolute()
DALI_DATA_PATH = ROOT / "data" / "dali"
DALI_AUDIO_PATH = ROOT / "data" / "audio"

dali_data = dali_code.get_the_DALI_dataset(DALI_DATA_PATH, skip=[], keep=[])
# dali_info = dali_code.get_info(DALI_DATA_PATH / 'info' / 'DALI_DATA_INFO.gz')


@dataclass
class Chunk:
    song_id: str
    init_sample: int
    end_sample: int
    init_time: float
    end_time: float
    lyrics: str

class DaliDataset(Dataset):
    def __init__(self, root_dir, chunk_size=16000, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.dali_data_path = self.root_dir / "dali"
        self.dali_audio_path = self.root_dir / "audio"
        self._load_data()
        self.chunk_size = chunk_size
        self.chunk_map: t.List[Chunk] = []
        self.dali_data = {}
        self.transform = transform

    def _load_data(self):

        ground_truth = dali_code.utilities.read_gzip(self.root_dir / "gt_v1.0_22_11_18.gz")
        dali_data = dali_code.get_the_DALI_dataset(DALI_DATA_PATH, ground_truth, skip=[], keep=[])
        # dali_info = dali_code.get_info(DALI_DATA_PATH / 'info' / 'DALI_DATA_INFO.gz')

        for file in DALI_AUDIO_PATH.glob("*.mp3"):
            iden = file.stem
            entry = dali_data[iden]
            entry.info["audio"]["path"] = file.absolute()
            entry.info["audio"]["metadata"] = torchaudio.info(file.absolute())
            if entry.info["metadata"]["language"] == "english":
                self.dali_data[iden] = entry

                notes = iter(entry.annotations["annot"]["notes"])
                initial_window = []
                for c in range(int(dali_data.info["audio"]["metadata"].num_samples / self.chunk_size)):
                    init_sample = c * self.chunk_size
                    end_sample = (c + 1) * self.chunk_size-1
                    initial_window.append([init_sample, end_sample])

                for chunk, idx in enumerate(initial_window):
                    init_time = frame2time(chunk[0], entry.info["audio"]["metadata"].sample_rate)
                    end_time =  frame2time(end_sample, entry.info["audio"]["metadata"].sample_rate)


                    valid_notes = []
                    note = "START"
                    while end_time > note["time"][1] or note != "END":
                        if note is None or note == "START":
                            try:
                                note = next(notes)
                            except StopIteration:
                                note = "END"
                        if note["time"][1] < end_time:
                            valid_notes.append(note)
                            note = None
                        else:
                            # set init on the next frame
                    


                    self.chunk_map.append(Chunk(iden, init_sample, end_sample, init_time, end_time))

    def __len__(self):
        return len(self.chunk_map)

    def __getitem__(self, idx):
        chunk_meta = self.chunk_map[idx]

        entry = self.dali_data[chunk_meta.song_id]
        waveform, sample_rate = torchaudio.load(
            entry.info["audio"]["path"],
            frame_offset=chunk_meta.init_sample,
            num_frames=self.chunk_size,
        )
        
        
        init_chunk_time = frame2time(chunk_meta.init_sample, sample_rate)
        end_chunk_time =  frame2time(chunk_meta.end_sample, sample_rate)

        

        return {"waveform": waveform, "lyrics": None}
