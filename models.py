import norse.torch as snn
import torch.nn as nn
import torch
import numpy as np


class Line1(torch.nn.Module):
    def __init__(self):
        """
        - LI 0.002
        """
        super(Line1, self).__init__()
        self.dt = 361e-6
        self.l1 = snn.LICell(
            p=snn.LIParameters(
                tau_mem_inv=1/0.001,
                tau_syn_inv=1/0.001,
                # v_leak=0,
            )
        )

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]  # .view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            outputs += [z]

        return torch.stack(outputs)


class Line2(torch.nn.Module):
    def __init__(self):
        """
        - LIF 0.001
        - LI 0.002
        """
        super(Line2, self).__init__()
        self.dt = 361e-6
        self.l1 = snn.LICell(p=snn.LIParameters(tau_mem_inv=1/0.001))
        self.l2 = snn.LICell(p=snn.LIParameters(tau_mem_inv=1/0.001))

        kernel_size = 9
        kernel = torch.zeros(kernel_size, kernel_size,)
        if torch.cuda.is_available():
            kernal = kernel.cuda(0)
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
        convolution.weight = torch.nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = s2 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]  # .view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.convolution(z)
            # z, s2 = self.l1(z, s2)
            outputs += [z]

        return torch.stack(outputs)


class Blur1(torch.nn.Module):
    def __init__(self):
        """
        - LIF 0.001
        - LI 0.002
        """
        super(Blur1, self).__init__()
        self.l1 = snn.LIFCell(p=snn.LIFParameters(
            tau_syn_inv=100, tau_mem_inv=400))
        self.l2 = snn.LICell(p=snn.LIFParameters(
            tau_syn_inv=800, tau_mem_inv=200))

        kernel = torch.tensor([[0,      0.15,   0],
                               [0.15,   0.4,    0.15],
                               [0,      0.15,   0]])
        # if torch.cuda.is_available():
        #     kernal = kernel.cuda(0)
        convolution = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        convolution.weight = torch.nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0))
        self.convolution = convolution

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = s2 = None
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]
            z = self.convolution(z)
            # z = self.convolution(z)
            z, s1 = self.l1(z, s1)
            # z, s2 = self.l2(z, s2)
            outputs += [z]

        return torch.stack(outputs)


@torch.jit.script
def _fast_lif_step_jit(input_tensor, state, decay_rate, threshold):
    state *= decay_rate
    state += input_tensor
    spikes = torch.gt(state, threshold).to(state.dtype)
    state -= spikes
    return spikes, state


class FastLIFCell(torch.nn.Module):
    def __init__(self, decay_rate, threshold) -> None:
        super().__init__()
        self.decay_rate = torch.tensor(decay_rate)
        self.threshold = torch.tensor(threshold)

    def forward(self, input_tensor, state):
        return _fast_lif_step_jit(input_tensor, state, self.decay_rate, self.threshold)


class Fast1(torch.nn.Module):
    def __init__(self):
        """
        """
        super(Fast1, self).__init__()
        self.l1 = FastLIFCell(0.5, 2)

        kernel = torch.tensor([[0,      0.15,   0],
                               [0.15,   0.4,    0.15],
                               [0,      0.15,   0]])
        # Google how to put whole model on CUDA
        # if torch.cuda.is_available():
        #     kernel = kernel.cuda(0)
        convolution = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        convolution.weight = torch.nn.Parameter(
            kernel.unsqueeze(0).unsqueeze(0))
        self.convolution = convolution

    def forward(self, x):
        seq_length, _, _, _ = x.shape
        s1 = torch.zeros(x.shape[1:], device=x.device,dtype=torch.float32)
        outputs = []

        for ts in range(seq_length):
            z = x[ts, :, :, :]
            z = self.convolution(z)
            z, s1 = self.l1(z, s1)
            outputs += [z]

        return torch.stack(outputs)