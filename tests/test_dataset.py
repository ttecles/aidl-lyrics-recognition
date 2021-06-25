import pathlib
from unittest import TestCase
from unittest.mock import patch

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
            ann.info["id"] = "test"
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
                      audio_start=0, audio_end=1 * chunk_length - 1,
                      lyrics='I AM'),
                Chunk(song_id='test',
                      init_sample=1 * chunk_length, end_sample=2 * chunk_length - 1,
                      audio_start=1 * chunk_length, audio_end=1099,
                      lyrics='AN'),
                Chunk(song_id='test',
                      init_sample=2 * chunk_length, end_sample=3 * chunk_length - 1,
                      audio_start=1701, audio_end=1147 * 2 - 1,
                      lyrics='MAZING')
            ])

            self.assertEqual(tuple(dt[0][0].size()), (2, chunk_length))
            self.assertEqual(tuple(dt[1][0].size()), (2, chunk_length))
            self.assertEqual(tuple(dt[2][0].size()), (2, chunk_length))

    def test_notes_bigger_than_chunk(self):
        self.maxDiff = None
        with patch("lyre.data.dali_code") as mock_dali:
            # length of our test audio: 1147  -  2294
            # sample of our test audio: 22050 - 44100
            ann = Annotations()
            ann.info["id"] = "test"
            ann.info["metadata"]["language"] = "english"
            ann.annotations["annot"]["notes"] = [
                {"time": (sample2time(200, 44100), sample2time(2000, 44100)), "text": "I"}]
            ann.annotations["annot"]["paragraphs"] = [{"text": "I"}]
            mock_dali.get_the_DALI_dataset.return_value = {"test": ann}

            chunk_length = 800
            dt = DaliDataset(pathlib.Path("."), length=chunk_length)

            self.assertListEqual(dt.chunk_map, [
                Chunk(song_id='test',
                      init_sample=0 * chunk_length, end_sample=1 * chunk_length - 1,
                      audio_start=0, audio_end=199,
                      lyrics=''),
                Chunk(song_id='test',
                      init_sample=2 * chunk_length, end_sample=3 * chunk_length - 1,
                      audio_start=2001, audio_end=1147 * 2 - 1,
                      lyrics='')
            ])

    # Chunk(song_id='857676094d814db2b368a49759a3fe5b', init_sample=0, end_sample=440999, audio_start=-8600,
    #       audio_end=440999,
    #       lyrics='TWO, THREE, FOUR. SHAPES MADE OF FOUR COLORED BLOCKS LIKE A T OR A BOX COME DOWN LIKE FALLING BRICKS. YOU CAN PLACE THEM IN ROWS BUT EVERYBODY KNOWS')
