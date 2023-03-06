#%%
import numpy as np
import glob
import tonic
import os
import sys
from metavision_core.event_io import load_events

# %% dat2npy events

os.chdir(sys.path[0])

for f in glob.glob("./data/Line/*.dat"):
    print(f)
    np.save(f[0:-4], load_events(f))


# %% dat2npy frames
os.chdir(sys.path[0])
transform = tonic.transforms.ToFrame(
    sensor_size=[480, 360, 1],
    time_window=361,
)
for f in glob.glob("./data/Line/*.dat"):
    print(f)
    np.save(f[0:-4]+"_frames", transform(load_events(f)))
# %%
