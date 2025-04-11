import torch
from gw_simulator.generate import run_sim
from gw_simulator.simulator.interface import BilbyGravitationalWaveBenchmarkSimulator as gws

def test_simple_sim():

    batchsize = 1
    theta = torch.ones((2,))*20.

    xs = run_sim(theta)

    assert xs.shape != theta.shape
    assert list(xs.shape) == [1, 3, 8192]


def test_small_sim():

    batchsize = 1
    theta = torch.ones((2,))*20.

    sim = gws(sampling_frequency=1024)
    xs = run_sim(theta, simulator=sim)

    print(xs.shape)
    assert xs.shape != theta.shape
    assert list(xs.shape) == [1, 3, 8192//2]


def test_batched_sim():

    batchsize = 4
    theta = torch.ones((batchsize, 2)) * 20.
    theta[..., 0] += 2*torch.arange(1, batchsize+1)


    xs = run_sim(theta)

    assert xs.shape != theta.shape
    assert list(xs.shape) == [batchsize, 3, 8192]

    obs = xs[0,2]
    exp = xs[1,2]
    assert not torch.allclose(obs, exp, atol=1e-22)
