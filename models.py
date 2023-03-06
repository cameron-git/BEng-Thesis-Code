import norse.torch as snn
import torch.nn as nn
import torch
import numpy as np

kernel_size = 9
kernel = torch.zeros(kernel_size, kernel_size)
kernel[int((kernel_size - 1) / 2), :] = 1
# plt.matshow(kernel.T)
convolution = torch.nn.Conv2d(
    1,
    1,
    kernel_size,
    padding=4,
    bias=False,
    dilation=1,
)
convolution.weight = torch.nn.Parameter(kernel.unsqueeze(0).unsqueeze(0))

class SNN1(torch.nn.Module):
    def __init__(self):
        """
        - LI 0.002
        """
        super(SNN1, self).__init__()
        self.dt = 361e-6
        self.l1 = snn.LICell(p=snn.LIParameters(tau_mem_inv=1/0.002))

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]  # .view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            outputs += [z]

        return torch.stack(outputs)


class SNN2(torch.nn.Module):
    def __init__(self):
        """
        - LIF 0.001
        - LI 0.002
        """
        super(SNN2, self).__init__()
        self.dt = 361e-6
        self.l1 = snn.LICell(p=snn.LIParameters(tau_mem_inv=1/0.001))
        self.l2 = snn.LICell(p=snn.LIParameters(tau_mem_inv=1/0.001))

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = s2 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]  # .view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = convolution(z)
            z, s2 = self.l1(z, s2)
            outputs += [z]

        return torch.stack(outputs)
