import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser.make_state import load_batch
from imitator.make_layers import *
from imitator.train import ConvNet
from imitator.use_model import find_best, find_probs

# Uses model that predicts expert othello player moves to create a heatmap over probabilities of next actions

MODEL_PATH = './imitator/model_saves/imitator_3.5.pth'
TEST_PATH = './parser/test.txt'
BATCH_SIZE = 1

class Imitator():
    def __init__(self):
        super().__init__()
        self.model = ConvNet()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    def choose_move(self, state): # state given
        self.model.eval()
        return find_best(self.model, state[0])
