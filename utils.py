import numpy as np
import tonic
import torch


def n_evs(evs):
    """
    MNIST: 4172
    POKERDVS: 2991
    DVSGesture: 361903
    ASLDVS: 28149
    CIFAR10DVS: 204409
    """
    sum = 0
    for i, ev in enumerate(evs):
        # print(i)
        sum = sum + (ev[0].size-sum)/(i+1)
    print(sum)


def t_evs(evs):
    """
    MNIST: 307663
    POKERDVS: 17283
    DVSGesture: 6455240
    ASLDVS: 110247
    CIFAR10DVS: 1293438
    """
    sum = 0
    for i, ev in enumerate(evs):
        # print(i)
        sum = sum + (ev[0][-1][0]-sum)/(i+1)
    print(sum)


def frame_merge(in_frames, n):
    """
    n: number of frames per merge
    """
    shape = list(in_frames.shape)
    shape[0] = int(shape[0]/n)
    out_frames = torch.empty(shape)
    for i in range(shape[0]):
        j = i*n
        out_frames[i] = torch.sum(in_frames[j:j+n], 0)
    return out_frames
