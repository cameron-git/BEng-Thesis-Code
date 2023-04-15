import norse.torch as snn
import torch
import torch.nn as nn
from typing import NamedTuple

class LIFParametersJIT(NamedTuple):
    """Parametrization of a LIF neuron

    Parameters:
        tau_syn_inv (torch.Tensor): inverse synaptic time
                                    constant (:math:`1/\\tau_\\text{syn}`) in 1/ms
        tau_mem_inv (torch.Tensor): inverse membrane time
                                    constant (:math:`1/\\tau_\\text{mem}`) in 1/ms
        v_leak (torch.Tensor): leak potential in mV
        v_th (torch.Tensor): threshold potential in mV
        v_reset (torch.Tensor): reset potential in mV
        method (str): method to determine the spike threshold
                      (relevant for surrogate gradients)
        alpha (torch.Tensor): hyper parameter to use in surrogate gradient computation
    """

    tau_syn_inv: torch.Tensor
    tau_mem_inv: torch.Tensor
    v_leak: torch.Tensor
    v_th: torch.Tensor
    v_reset: torch.Tensor
    method: str
    alpha: torch.Tensor

class IFCell(snn.module.snn.SNNCell):


    def __init__(self, p = snn.LIFParameters(), **kwargs):
        super().__init__(
            activation=if_feed_forward_step,
            state_fallback=self.initial_state,
            p=snn.LIFParameters(
                torch.as_tensor(p.tau_syn_inv),
                torch.as_tensor(p.tau_mem_inv),
                torch.as_tensor(p.v_leak),
                torch.as_tensor(p.v_th),
                torch.as_tensor(p.v_reset),
                p.method,
                torch.as_tensor(p.alpha),
            ),
            **kwargs,
        )

    def initial_state(self, input_tensor: torch.Tensor):
        state = snn.LIFFeedForwardState(
            v=torch.full(
                input_tensor.shape,
                torch.as_tensor(self.p.v_leak).detach(),
                device=input_tensor.device,
                dtype=torch.float32,
            ),
            i=torch.zeros(
                *input_tensor.shape,
                device=input_tensor.device,
                dtype=torch.float32,
            ),
        )
        state.v.requires_grad = True
        return state
    
def if_feed_forward_step(
    input_spikes: torch.Tensor,
    state: snn.LIFFeedForwardState,
    p = snn.LIFParameters(),
    dt: float = 0.001,
):
    z, state = _if_feed_forward_step_jit(input_spikes, state, LIFParametersJIT(*p), dt)
    return z, state

@torch.jit.ignore
def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return SuperSpike.apply(x, alpha)

@torch.jit.script
def _if_feed_forward_step_jit(
    input_tensor: torch.Tensor,
    state: snn.LIFFeedForwardState,
    p: LIFParametersJIT,
    dt: float = 0.001,
) :  # pragma: no cover
    # compute voltage updates
    dv = dt * p.tau_mem_inv * state.i
    v_decayed = state.v + dv

    # compute current updates
    di = -dt * p.tau_syn_inv * state.i
    i_decayed = state.i + di

    # compute new spikes
    z_new = super_fn(v_decayed - p.v_th, p.alpha)
    # compute reset
    v_new = (1 - z_new) * v_decayed + z_new * p.v_reset
    # compute current jumps
    i_new = i_decayed + input_tensor

    return z_new, snn.LIFFeedForwardState(v=v_new, i=i_new)

from norse.torch.functional.heaviside import heaviside


class SuperSpike(torch.autograd.Function):
    @staticmethod
    @torch.jit.ignore
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return heaviside(input_tensor)

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(
            2
        )  # section 3.3.2 (beta -> alpha)
        return grad, None


@torch.jit.ignore
def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return SuperSpike.apply(x, alpha)