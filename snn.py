# %% Import packages
from matplotlib import pyplot as plt  # graphic library, for plots
import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import tonic
import norse

# %% Import classes
from metavision_core.event_io import EventsIterator
from metavision_core.event_io.raw_reader import RawReader

# %% Import data
raw_path = "./data/1hzplane.raw"

# Creates an iterator. Faster
event_iterator = EventsIterator(
    raw_path,
    delta_t=1,
    max_duration=int(1e2),
)
print(event_iterator)
print("Imager size : ", event_iterator.get_size())

# Creates a buffer and then can be loaded as numpy
record_raw = RawReader(raw_path)
print("\n" + str(record_raw))
print("Imager size : ", record_raw.get_size())


# %%
def ev_rate_computation_iterator():
    # we preallocate an array for the result
    ev_rate_millisecond = np.zeros(int(1e6) // 1000)
    for ev in EventsIterator(raw_path, delta_t=10000, max_duration=int(1e6)):
        # np.unique allow to do an histogram quickly
        index, counts = np.unique(ev['t'] // 1000, return_counts=True)
        # for each timestamp (in millisecond) in index, we get the number of events in counts
        ev_rate_millisecond[index.astype(int)] = counts
    return ev_rate_millisecond


ev_rate_millisecond = ev_rate_computation_iterator()

plt.plot(np.arange(int(1e6) // 1000) * 1000, ev_rate_millisecond)
plt.title('Number of events by milliseconds as a function of time in us')
plt.show()


# %%
def viz_events(events, height, width):
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    img[events['y'], events['x']] = 255 * events['p'][:, None]
    return img


height, width = record_raw.get_size()

# load the next 50 ms worth of events
events = record_raw.load_delta_t(50000)
im = viz_events(events, height, width)

plt.imshow(im)
plt.tight_layout()

# %%
kernel_size = 5
kernel = torch.zeros(5, 5)
kernel[2, :] = torch.ones(5)
plt.matshow(kernel)
convolution = torch.nn.Conv2d(1, 1, kernel_size)
convolution.weight = torch.nn.Parameter(kernel)

# %%
# Create Norse network
# - One refractory cell to inhibit pixels
# - One convolutional edge-detection layer
net = norse.torch.SequentialState(
    norse.torch.LIFRefracCell(),
    convolution,
)
state = None  # Start with empty state

# %%
import sdl2dll
from sdl2 import create_sdl_surface, events_to_bw

window, pixels = create_sdl_surface(640 * 2, 480)

# %%
for tensor in event_iterator:
    print(tensor)
    with torch.inference_mode():
        filtered, state = net(tensor.view(1, 1, 640, 480), state)

# %%
# from norse.torch import LIFParameters, LIFState
# from norse.torch.module.lif import LIFCell, LIFRecurrentCell
# from norse.torch import LICell, LIState

# from typing import NamedTuple

# class SNNState(NamedTuple):
#     lif0: LIFState
#     readout: LIState

# class SNN(torch.nn.Module):
#     def __init__(
#         self,
#         input_features,
#         hidden_features,
#         output_features,
#         tau_syn_inv,
#         tau_mem_inv,
#         record=False,
#         dt=1e-3,
#     ):
#         super(SNN, self).__init__()
#         self.l1 = LIFRecurrentCell(
#             input_features,
#             hidden_features,
#             p=LIFParameters(
#                 alpha=100,
#                 v_th=torch.as_tensor(0.3),
#                 tau_syn_inv=tau_syn_inv,
#                 tau_mem_inv=tau_mem_inv,
#             ),
#             dt=dt,
#         )
#         self.input_features = input_features
#         self.fc_out = torch.nn.Linear(hidden_features,
#                                       output_features,
#                                       bias=False)
#         self.out = LICell(dt=dt)

#         self.hidden_features = hidden_features
#         self.output_features = output_features
#         self.record = record

#     def forward(self, x):
#         seq_length, batch_size, _, _, _ = x.shape
#         s1 = so = None
#         voltages = []

#         if self.record:
#             self.recording = SNNState(
#                 LIFState(
#                     z=torch.zeros(seq_length, batch_size,
#                                   self.hidden_features),
#                     v=torch.zeros(seq_length, batch_size,
#                                   self.hidden_features),
#                     i=torch.zeros(seq_length, batch_size,
#                                   self.hidden_features),
#                 ),
#                 LIState(
#                     v=torch.zeros(seq_length, batch_size,
#                                   self.output_features),
#                     i=torch.zeros(seq_length, batch_size,
#                                   self.output_features),
#                 ),
#             )

#         for ts in range(seq_length):
#             z = x[ts, :, :, :].view(-1, self.input_features)
#             z, s1 = self.l1(z, s1)
#             z = self.fc_out(z)
#             vo, so = self.out(z, so)
#             if self.record:
#                 self.recording.lif0.z[ts, :] = s1.z
#                 self.recording.lif0.v[ts, :] = s1.v
#                 self.recording.lif0.i[ts, :] = s1.i
#                 self.recording.readout.v[ts, :] = so.v
#                 self.recording.readout.i[ts, :] = so.i
#             voltages += [vo]

#         return torch.stack(voltages)

# # %%
# LR = 0.002
# INPUT_FEATURES = np.product(trainset.sensor_size)
# HIDDEN_FEATURES = 100
# OUTPUT_FEATURES = len(trainset.classes)

# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")

# model = Model(
#     snn=SNN(
#         input_features=INPUT_FEATURES,
#         hidden_features=HIDDEN_FEATURES,
#         output_features=OUTPUT_FEATURES,
#         tau_syn_inv=torch.tensor(1 / 1e-2),
#         tau_mem_inv=torch.tensor(1 / 1e-2),
#     ),
#     decoder=decode,
# ).to(DEVICE)

# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# model
