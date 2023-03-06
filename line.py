# %% Imports
# %matplotlib ipympl
%matplotlib tk
import models
import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tonic
from metavision_core.event_io import EventsIterator, load_events
# import sdl2.ext

# %% Dataset

file_name = "1hzplane"

# file = load_events("./data/Line/" + file_name + ".dat")
# file = np.load("./data/Line/" + file_name + ".npy")

in_frames = torch.from_numpy(
    np.load("./data/Line/"+file_name+"_frames.npy")[0:2000,:,:]
    ).cuda(0)

# data_iterator = EventsIterator(
#     file_path,
#     delta_t=361,
#     n_events=1,
#     # mode="n_events",
#     # max_duration=100000,
# )

# %% Modeltorch

net = models.SNN2()

# %% Training

# %% Inference

with torch.inference_mode():
    out_frames = net(in_frames)

# np.save("./data/Line/" + file_name + "_out.npy")

# %% Visualise

animation = tonic.utils.plot_animation(frames=out_frames.cpu())

animation = tonic.utils.plot_animation(frames=in_frames.cpu())

# %%
