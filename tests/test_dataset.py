import pathlib
from unittest import TestCase

from lyre.data import DaliDataset

root = pathlib.Path(__file__).parent.parent

class TestDaliDataset(TestCase):

    def test_creation_of_dataset(self):
        dt = DaliDataset(root / "data", length=10000)
