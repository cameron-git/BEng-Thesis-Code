# %% Imports
from metavision_core.event_io import EventsIterator, load_events
import tonic
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
import torch
import models
import random
%matplotlib inline
# %matplotlib ipympl
# %matplotlib tk

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

# i_evs = tonic.datasets.NMNIST(save_to="./data")
# i_evs = tonic.datasets.POKERDVS(save_to="./data")
# i_evs = tonic.datasets.DVSGesture(save_to="./data")
# i_evs = tonic.datasets.ASLDVS(save_to="./data")
i_evs = tonic.datasets.CIFAR10DVS(save_to="./data")

# %%

to_frame = tonic.transforms.ToFrame(i_evs.sensor_size, time_window=10000)

events, label = i_evs[random.randint(0,59999)]
print(label)
tonic.utils.plot_animation(to_frame(events))
plt.show()


# %% Mean events per sample

def n_evs(evs):
    sum = 0
    for i, ev in enumerate(evs):
        # print(i)
        sum = sum + (ev[0].size-sum)/(i+1)
    print(sum)
# MNIST: 4172
# POKERDVS: 2991
# DVSGesture: 361903
# ASLDVS: 
# CIFAR10DVS: 

# %%


to_frame = tonic.transforms.ToFrame(
    sensor_size=i_evs.sensor_size,
)
noise = tonic.transforms.UniformNoise(
    sensor_size=i_evs.sensor_size,
    n=n_evs(i_evs)*0.1,
)

# %%

animation = tonic.utils.plot_animation(frames=ifs)
animation = tonic.utils.plot_animation(frames=ofs)

# %% Modeltorch

net = models.SNN2()

# %% Training

# %% Inference

with torch.inference_mode():
    ofs = net(ifs)

# np.save("./data/Line/" + file_name + "_out.npy")

# %% Visualise

animation = tonic.utils.plot_animation(frames=ofs.cpu())

# %%
