import torch


@torch.jit.script
def _fast_lif_step_jit(input_tensor, state, decay_rate, threshold):
    state *= decay_rate
    state += input_tensor
    spikes = torch.gt(state, threshold).to(state.dtype)
    state -= spikes * threshold
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


@torch.jit.script
def _fast_if_step_jit(input_tensor, state, threshold):
    state += input_tensor
    spikes = torch.gt(state, threshold).to(state.dtype)
    state -= spikes * threshold
    return spikes, state

class FastIFCell(torch.nn.Module):
    def __init__(self,  threshold) -> None:
        super().__init__()
        self.threshold = torch.tensor(threshold)

    def forward(self, input_tensor, state):
        return _fast_if_step_jit(input_tensor, state, self.threshold)