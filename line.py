# %% Imports
# %matplotlib ipympl
%matplotlib tk
from metavision_core.event_io import EventsIterator, load_events

import tonic
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import torch
# import sdl2.ext

# %% Dataset

file_name = "1hzplane"

file = load_events("./data/Line/" + file_name + ".dat")
# file = np.load("./data/Line/" + file_name + ".npy")

# data_iterator = EventsIterator(
#     file_path,
#     delta_t=361,
#     n_events=1,
#     # mode="n_events",
#     # max_duration=100000,
# )

# %% Model

import models

net = models.SNN2()

# %% Training

# %% Inference

in_frames = np.load("./data/Line/"+file_name+"_frames.npy")
in_frames = torch.from_numpy(in_frames)

# %%


# %%
with torch.inference_mode():
    out_frames = net(in_frames)


# np.save("./data/Line/" + file_name + "_out.npy")

# %% Visualise

animation = tonic.utils.plot_animation(frames=out_frames)

# %%
