import numpy as np
import torch
import torch.nn as nn

from .data import DaliDataset
from .model import DemucsWav2Vec
from .train import train_model
from .train import test_model

Mydataset = DaliDataset()
Mymodel = DemucsWav2Vec()

criterion = nn.CTCLoss()
optimizer = torch.optim.Adam(Mymodel.parameters(), lr=0.0001)
epochs = 100


train_model(Mymodel, optimizer, criterion, epochs)

test_model(test_data, Mymodel, criterion)