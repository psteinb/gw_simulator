from gw_simulator.simulator.util import UniformMassPrior

def test_prior_samples():

    batchsize = 4
    prior = UniformMassPrior(20, 30)
    obs = prior.sample((batchsize,))
    print(obs)
    assert obs.shape == (batchsize, 2)
    assert obs.max() < 31.
    assert obs.min() > 19.
