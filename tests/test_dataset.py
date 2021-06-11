import pathlib
from unittest import TestCase
from unittest.mock import patch, Mock, PropertyMock

from DALI import Annotations

from lyre.data import DaliDataset, Chunk
from lyre.utils import sample2time


class TestDaliDataset(TestCase):


    def test_creation_of_dataset(self):
        self.maxDiff = None
        with patch("lyre.data.dali_code") as mock_dali:
            # length of our test audio: 1147
            # sample of our test audio: 22050
            ann = Annotations()
            ann.info["metadata"]["language"] = "english"
            ann.annotations["annot"]["notes"] = [
                {"time": (sample2time(200, 44100), sample2time(500, 44100)), "text": "I"},
                {"time": (sample2time(600, 44100), sample2time(700, 44100)), "text": "am"},
                {"time": (sample2time(900, 44100), sample2time(1000, 44100)), "text": "an"},
                {"time": (sample2time(1100, 44100), sample2time(1700, 44100)), "text": "a"},
                {"time": (sample2time(1800, 44100), sample2time(2000, 44100)), "text": "mazing"}]
            ann.annotations["annot"]["paragraphs"] = [{"text": "I am an amazing"}]
            mock_dali.get_the_DALI_dataset.return_value = {"test": ann}

            chunk_length = 800
            dt = DaliDataset(pathlib.Path("."), length=chunk_length)

            self.assertListEqual(dt.chunk_map, [
                Chunk(song_id='test',
                      init_sample=0 * chunk_length, end_sample=1 * chunk_length - 1,
                      audio_start=0,  audio_end=1 * chunk_length - 1,
                      lyrics='I am'),
                Chunk(song_id='test',
                      init_sample=1 * chunk_length, end_sample=2 * chunk_length - 1,
                      audio_start=1 * chunk_length, audio_end=1099,
                      lyrics='an'),
                Chunk(song_id='test',
                      init_sample=2 * chunk_length, end_sample=3 * chunk_length - 1,
                      audio_start=1701, audio_end=1147 * 2 - 1,
                      lyrics='mazing')
            ])

            self.assertEqual(tuple(dt[0]['waveform'].size()), (2, chunk_length))
            self.assertEqual(tuple(dt[1]['waveform'].size()), (2, chunk_length))
            self.assertEqual(tuple(dt[2]['waveform'].size()), (2, chunk_length))
