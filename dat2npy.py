import numpy as np
import glob
import os
import sys
from metavision_core.event_io import load_events

os.chdir(sys.path[0])

for f in glob.glob("./data/Line/*.dat"):
    print(f)
    np.save(f[0:-4], load_events(f))