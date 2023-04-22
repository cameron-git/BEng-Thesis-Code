import norse.torch as snn
import torch.nn as nn
import torch
import numpy as np
import tonic
from models.if_cell import *
from models.fast import *

from typing import NamedTuple


class kernels():
    plus3 = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])/5
    plus5 = np.array([[0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0],
                      [1, 1, 1, 1, 1],
                      [0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0]])/9
    gaussian3 = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]])/16
    gaussian5 = np.array([[1, 4,  7,  4,  1],
                          [4, 16, 26, 16, 4],
                          [7, 26, 41, 26, 7],
                          [4, 16, 26, 16, 4],
                          [1, 4,  7,  4,  1]])/273
    line3 = np.array([[0, 0, 0],
                      [1, 1, 1],
                      [0, 0, 0]])/3
    line5 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])/5
    linegaussian3 = np.array([[1, 1, 1],
                              [2, 2, 2,],
                              [1, 1, 1]])/12
    linegaussian5 = np.array([[7, 7, 7, 7, 7],
                              [26, 26, 26, 26, 26],
                              [41, 41, 41, 41, 41],
                              [26, 26, 26, 26, 26],
                              [7, 7, 7, 7, 7]])/535


class LIF(torch.nn.Module):
    def __init__(self):
        """
        - LI 0.002
        """
        super(LIF, self).__init__()
        self.dt = 361e-6
        self.tau_syn_inv = nn.Parameter(torch.as_tensor(1 / 0.01))
        self.tau_mem_inv = nn.Parameter(torch.as_tensor(1 / 0.01))
        self.tau_syn_inv.requires_grad
        self.tau_mem_inv.requires_grad
        self.l1 = snn.LIFCell(p=snn.LIFParameters(
            self.tau_syn_inv, self.tau_mem_inv))

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]
            z, s1 = self.l1(z, s1)
            outputs += [z]

        return torch.stack(outputs)


class FastLIF(torch.nn.Module):
    def __init__(self, decay, threshold):
        super(FastLIF, self).__init__()
        self.dt = 361e-6
        self.l1 = FastLIFCell(decay, threshold)

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = torch.zeros(x.shape[1:], device=x.device, dtype=torch.float32)
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]  # .view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            outputs += [z]

        return torch.stack(outputs)


class Conv_LI(torch.nn.Module):
    def __init__(self, kernel):
        super(Conv_LI, self).__init__()
        self.dt = 361e-6
        self.l1 = snn.LICell(p=snn.LIParameters(tau_mem_inv=1/0.001))
        kernel = torch.tensor(kernel).float()
        # kernal = kernel.cuda(0)
        convolution = torch.nn.Conv2d(
            1,
            1,
            kernel.shape[0],
            padding=int((kernel.shape[0]-1)/2),
            bias=False,
        )
        convolution.weight = torch.nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]  # .view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.convolution(z)
            outputs += [z]

        return torch.stack(outputs)


class Conv_LIF(torch.nn.Module):
    def __init__(self, kernel):
        super(Conv_LIF, self).__init__()
        self.l1 = snn.LIFCell(p=snn.LIFParameters(
            tau_syn_inv=torch.as_tensor(1 / 0.01),
        ))
        kernel = torch.tensor(kernel).float()
        # if torch.cuda.is_available():
        #     kernal = kernel.cuda(0)
        convolution = torch.nn.Conv2d(
            1,
            1,
            kernel.shape[0],
            padding=int((kernel.shape[0]-1)/2),
            bias=False,
        )
        convolution.weight = torch.nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0))
        self.convolution = convolution

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]
            z = self.convolution(z)
            z, s1 = self.l1(z, s1)
            outputs += [z]

        return torch.stack(outputs)


class LIF_LI(torch.nn.Module):
    def __init__(self):
        """
        """
        super(LIF_LI, self).__init__()
        self.l1 = snn.LIFCell(p=snn.LIFParameters(
            tau_syn_inv=400, tau_mem_inv=200))
        self.l2 = snn.LICell(p=snn.LIParameters(
            tau_syn_inv=400, tau_mem_inv=200))

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = s2 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]
            z, s1 = self.l1(z, s1)
            z, s2 = self.l1(z, s2)
            outputs += [z]

        return torch.stack(outputs)


class Denoise(torch.nn.Module):
    def __init__(self, t):
        """
        """
        super(Denoise, self).__init__()
        self.denoise = tonic.transforms.Denoise(t)

    def forward(self, x):
        return self.denoise(x)


class Conv_FastLIF(torch.nn.Module):
    def __init__(self, kernel, decay, threshold):
        """
        """
        super(Conv_FastLIF, self).__init__()
        self.l1 = FastLIFCell(decay, threshold)
        kernel = torch.tensor(kernel).float()
        # kernel = kernel.cuda(0)
        convolution = torch.nn.Conv2d(
            1,
            1,
            kernel.shape[0],
            padding=int((kernel.shape[0]-1)/2),
            bias=False,
        )
        convolution.weight = torch.nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0))
        self.convolution = convolution

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = torch.zeros(x.shape[1:], device=x.device, dtype=torch.float32)
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]
            z = self.convolution(z)
            z, s1 = self.l1(z, s1)
            outputs += [z]

        return torch.stack(outputs)


class Conv_FastLIF_FastLIF(torch.nn.Module):
    def __init__(self, kernel):
        """
        """
        super(Conv_FastLIF_FastLIF, self).__init__()
        self.l1 = FastLIFCell(0.85, 2)
        self.l2 = FastLICell(0.9)
        kernel = torch.tensor(kernel).float()
        # kernel = kernel.cuda(0)
        convolution = torch.nn.Conv2d(
            1,
            1,
            kernel.shape[0],
            padding=int((kernel.shape[0]-1)/2),
            bias=False,
        )
        convolution.weight = torch.nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0))
        self.convolution = convolution

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = torch.zeros(x.shape[1:], device=x.device, dtype=torch.float32)
        s2 = torch.zeros(x.shape[1:], device=x.device, dtype=torch.float32)
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]
            z = self.convolution(z)
            z, s1 = self.l1(z, s1)
            z, s2 = self.l2(z, s2)
            outputs += [z]

        return torch.stack(outputs)
