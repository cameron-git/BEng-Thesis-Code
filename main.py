# %% Imports
# %matplotlib ipympl
from metavision_core.event_io import EventsIterator, load_events
import tonic
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import torch
import models
%matplotlib tk
# import sdl2.ext

# %% Dataset

file_name = "1hzplane"

# ievs = load_events("./data/Line/" + file_name + ".dat")
# ievs = np.load("./data/Line/" + file_name + ".npy")
# ifs = torch.from_numpy(
#     np.load("./data/Line/"+file_name+"_frames.npy")[0:2000,:,:]
#     )
# if torch.cuda.is_available():
#   ifs = ifs.cuda(0)


# %%

ifs = tonic.datasets.NMNIST(save_to="./data",
                                transform=tonic.transforms.ToFrame,)
ifs = tonic.datasets.POKERDVS(save_to="./data",
                                transform=tonic.transforms.ToFrame,)
ifs = tonic.datasets.DVSGesture(save_to="./data",
                                transform=tonic.transforms.ToFrame,)
ifs = tonic.datasets.ASLDVS(save_to="./data",
                                transform=tonic.transforms.ToFrame,)
ifs = tonic.datasets.CIFAR10DVS(save_to="./data",
                                transform=tonic.transforms.ToFrame,)


# %% Modeltorch

net = models.SNN2()

# %% Training

# %% Inference

with torch.inference_mode():
    ofs = net(ifs)

# np.save("./data/Line/" + file_name + "_out.npy")

# %% Visualise

animation = tonic.utils.plot_animation(frames=ofs.cpu())

animation = tonic.utils.plot_animation(frames=ifs.cpu())

# %%
