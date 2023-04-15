import norse.torch as snn
import torch.nn as nn
import torch
import numpy as np
import tonic


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
        0       0.15    0
        0.15    0.4     0.15
        0       0.15    0
        """
        super(Blur1, self).__init__()
        self.l1 = snn.LIFCell(p=snn.LIFParameters(
            tau_syn_inv=200, tau_mem_inv=200))
        self.l2 = snn.LICell(p=snn.LIParameters(
            tau_syn_inv=400, tau_mem_inv=200))

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
            z, s1 = self.l1(z, s1)
            # z, s2 = self.l2(z, s2)
            outputs += [z]

        return torch.stack(outputs)


class Blur2(torch.nn.Module):
    def __init__(self):
        """
        1/9 1/9 1/9
        1/9 1/9 1/9
        1/9 1/9 1/9
        """
        super(Blur2, self).__init__()
        self.l1 = snn.LIFCell(p=snn.LIFParameters(
            tau_syn_inv=400, tau_mem_inv=200))
        self.l2 = snn.LICell(p=snn.LIParameters(
            tau_syn_inv=400, tau_mem_inv=200))

        kernel = torch.full((3, 3), 1/9)
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
            z, s1 = self.l1(z, s1)
            # z, s2 = self.l2(z, s2)
            outputs += [z]

        return torch.stack(outputs)


class Blur3(torch.nn.Module):
    def __init__(self):
        """
        1/9 1/9 1/9
        1/9 1/9 1/9
        1/9 1/9 1/9
        """
        super(Blur2, self).__init__()
        self.l1 = snn.LIFCell(p=snn.LIFParameters(
            tau_syn_inv=400, tau_mem_inv=200))
        self.l2 = snn.LICell(p=snn.LIParameters(
            tau_syn_inv=400, tau_mem_inv=200))

        kernel = torch.full((3, 3), 1/9)
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
            z, s1 = self.l1(z, s1)
            # z, s2 = self.l2(z, s2)
            outputs += [z]

        return torch.stack(outputs)


class Denoise1(torch.nn.Module):
    def __init__(self):
        """
        """
        super(Denoise1, self).__init__()
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


class Denoise2(torch.nn.Module):
    def __init__(self, t):
        """
        """
        super(Denoise2, self).__init__()
        self.denoise = tonic.transforms.Denoise(t)

    def forward(self, x):
        return self.denoise(x)


@torch.jit.script
def _fast_lif_step_jit(input_tensor, state, decay_rate, threshold):
    state *= decay_rate
    state += input_tensor
    spikes = torch.gt(state, threshold).to(state.dtype)
    # state -= spikes
    return spikes, state


class FastLIFCell(torch.nn.Module):
    def __init__(self, decay_rate, threshold) -> None:
        super().__init__()
        self.decay_rate = torch.tensor(decay_rate)
        self.threshold = torch.tensor(threshold)

    def forward(self, input_tensor, state):
        return _fast_lif_step_jit(input_tensor, state, self.decay_rate, self.threshold)


@torch.jit.script
def _fast_li_step_jit(input_tensor, state, decay_rate):
    state *= decay_rate
    state += input_tensor
    return state, state.clone()


class FastLICell(torch.nn.Module):
    def __init__(self, decay_rate) -> None:
        super().__init__()
        self.decay_rate = torch.tensor(decay_rate)

    def forward(self, input_tensor, state):
        return _fast_li_step_jit(input_tensor, state, self.decay_rate)


class Fast1(torch.nn.Module):
    def __init__(self):
        """
        """
        super(Fast1, self).__init__()
        self.l1 = FastLIFCell(0.9, 2)

        kernel = torch.tensor([[0,      0.15,   0],
                               [0.15,   0.4,    0.15],
                               [0,      0.15,   0]])
        convolution = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
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


class Fast2(torch.nn.Module):
    def __init__(self):
        """
        """
        super(Fast2, self).__init__()
        self.l1 = FastLIFCell(0.9, 2)

        kernel = torch.tensor([[0,      0,      0,      0,      0],
                               [0,      0,      0,      0,      0],
                               [1/5,    1/5,    1/5,    1/5,    1/5],
                               [0,      0,      0,      0,      0],
                               [0,      0,      0,      0,      0]])
        convolution = torch.nn.Conv2d(1, 1, 5, padding=2, bias=False)
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

class Fast3(torch.nn.Module):
    def __init__(self):
        """
        """
        super(Fast3, self).__init__()
        self.l1 = FastLIFCell(0.9, 2)
        self.l2 = FastLIFCell(0.95,1)

        kernel = torch.tensor([[0,      0,      0,      0,      0],
                               [0,      0,      0,      0,      0],
                               [1/5,    1/5,    1/5,    1/5,    1/5],
                               [0,      0,      0,      0,      0],
                               [0,      0,      0,      0,      0]])
        convolution = torch.nn.Conv2d(1, 1, 5, padding=2, bias=False)
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
