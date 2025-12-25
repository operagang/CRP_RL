import torch
import torch.nn as nn
from model.decoder import Decoder
import argparse

class Model(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.decoder = Decoder(args).to(self.device)


    # cost length ll:log_softmax  sum of probability  pi:predicted tour
    def forward(self, x, max_retrievals):
        decoder_output = self.decoder(x, max_retrievals)
        cost, ll, reloc = decoder_output
        return cost, ll, reloc
