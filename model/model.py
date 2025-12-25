import torch
import torch.nn as nn
from model.decoder import Decoder
import argparse

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.decoder = Decoder(args).to(self.device)

    def forward(self, x, max_retrievals=None):
        decoder_output = self.decoder(x, max_retrievals)
        cost, ll = decoder_output
        return cost, ll
