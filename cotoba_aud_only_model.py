# -*- coding: utf-8 -*- 

import sys
import cv2
import numpy as np

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.fc_new = nn.Linear(257*5,512)
        self.out = nn.Linear(512,257*5)
    
    def forward(self, aud):
        aud = aud.view(-1,257*5)
        aud = self.fc_new(aud)
        out_audio = self.out(aud)
        #out_audio = aud.view(-1,257,5)
        return out_audio
    
