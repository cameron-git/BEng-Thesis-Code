# %% Imports
%matplotlib ipympl
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tonic
import norse
import sdl2.ext
from metavision_core.event_io import EventsIterator, load_events

# %% Dataset

file_path = "/data/1hzplane_td.dat"


# data_iterator = EventsIterator(
#     file_path,
#     delta_t=361,
#     n_events=1,
#     # mode="n_events",
#     # max_duration=100000,
# )

# %% Model

kernel_size = 9
kernel = torch.zeros(kernel_size, kernel_size)
kernel[:, int((kernel_size - 1) / 2)] = 1
plt.matshow(kernel.T)
kernels = torch.stack((kernel, kernel.flip(0)))
convolution = torch.nn.Conv2d(
    1,
    2,
    kernel_size,
    padding=4,
    bias=False,
    dilation=1,
)
convolution.weight = torch.nn.Parameter(kernels.unsqueeze(1))

net = norse.torch.SequentialState(
    norse.torch.LICell(p=norse.torch.LIParameters(tau_mem_inv=1/0.5)),
    convolution,
)
state = None  # Start with empty state

# %% Training

# %% Inference

transform = tonic.transforms.ToFrame(
    sensor_size=[480,360,1],
    time_window=361,
)

frames = transform(np.load("../data/Line/1hzplane_td.npy"))


animation = tonic.utils.plot_animation(frames=frames)

# %%
