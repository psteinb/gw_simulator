# code created by Manuel Glöckler
import os
import sys
import time
from functools import partial
from pathlib import Path

import click
import h5py
import hdf5plugin
import numpy as np
import torch
from torch import multiprocessing

from gw_simulator.simulator.interface import \
    BilbyGravitationalWaveBenchmarkSimulator as gws

#from gw_simulator.utils import get_git_describe

## DEFAULT VALUES ###############################
# legacy settings
if "NUM_SIMS" not in os.environ.keys():
    NUM_SIMS = 10_000
else:
    NUM_SIMS = int(os.environ["NUM_SIMS"])

DEFAULT_SIMULATOR = gws()

# TODO: meta data for identifying the simulation (failed if not run in a git repo)
# GIT_DESCRIPTION = get_git_describe()


def run_sim(theta: torch.Tensor,
            simulator: gws = DEFAULT_SIMULATOR,
            domain: str = "time") -> torch.Tensor:
    """
    Perform one simulation given simulator object and the simulator parameters
    theta. The thetas encode the mass of the blackhole (index 0 in last dimension).

    Parameters
    ----------
    theta : torch.Tensor
        simulator parameters, provided in batched format

    simulator : GravitationalWaveBenchmarkSimulator
        simulator object

    domain : string
        domain of the signal, possible values "time" or frequency

    Examples
    --------
    > theta
    torch.Tensor([[20,25]])
    > x = run_sim(theta)
    # ...
    > x.shape
    [1,2,2024]

    Returns
    -------
    LIGO spectra in batched format
    """

    xs = simulator(theta.to("cpu"))

    signal = xs["detector"]["time"]
    noise  = xs["noise"]["time"]

    return signal + noise


def sim_and_store(thetas: torch.Tensor,
                  batch_id: int,
                  seed_in_use: int,
                  num_sims: int,
                  num_workers: int,
                  output: Path,
                  message: str = ""):
    """
    perform simulations and store output in hdf5 file. This function uses
    multiprocessing internally to facilitate parallel execution.

    Parameters
    ----------
    thetas : torch.Tensor
        simulation parameters to simulate, these are expected to be batched and
        have shape [num_sims, 2]
    batch_id : int
        unique ID for this batch of simulations in a simulation campaign
    seed_in_use : int
        seed that was used for this session (will not be reset in this function)
    num_sims : int
        number of simulations to perform
    num_workers : int
        number of parallel workers to use
    output : Path
        path of output file
    message : str
        optional message for reproducing the simulation

    Returns
    -------
    int : 0 if successful

    """
    #preparing the simulator
    simulator_ = gws()
    run_sim_ = partial(run_sim, simulator=simulator_)
    idx = batch_id

    #simulate gravitational waves
    start = time.time()
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(run_sim_, thetas)

    #concat signals into one tensor
    xs = torch.concat(results)
    end1 = time.time()

    dur_sec = end1 - start
    print(
        f"{num_sims} simulated samples produced in {dur_sec:.2f} s ({dur_sec/float(num_sims):02.4f} sample/s)"
    )
    print("thetas:", thetas.shape, thetas.dtype)
    print("xs:", xs.shape, xs.dtype)

    outfile = output.replace("<batchindex>", f"{batch_id:02.0f}")
    with h5py.File(outfile, "w") as out5:
        out5_xs = out5.create_dataset(
            "xs",
            data=xs,
            chunks=True,
            **hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=10),
        )
        out5_thetas = out5.create_dataset(
            "thetas",
            data=thetas,
            chunks=True,
            **hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=10),
        )

        out5_xs.attrs["bilby_version"] = bilby.__version__
        out5_xs.attrs["seed"] = seed_in_use
        if message is not None and len(message) > 0:
            out5_xs.attrs["msg"] = message

        out5_thetas.attrs["seed"] = seed_in_use

    end2 = time.time()
    dur_io_sec = end2 - end1
    print(
        f"{num_sims} simulated samples written to {outfile} in {dur_io_sec:.2f} s ({dur_io_sec/float(num_sims):02.4f} sample/s)"
    )
    return 0


@click.command()
@click.option('-I', '--batch-id', default=1, type=int, help='batch index.')
@click.option('-n', '--num_sims', default=NUM_SIMS, type=int, help='number of simulations to perform')
@click.option('-j', '--num_workers', default=1, type=int, help='number of worker processes to use in parallel')
@click.option('-o', '--output', default="gws-<batchindex>.h5", type=click.Path(), help='output file location, please keep <batchindex> as placeholder')
@click.option('-m', '--message', default="", type=str, help='optional metadata message for reproducibility')
def main(batch_id, num_sims, num_workers, output, message):
    """ will generate {NUM_SIMS} samples from simulator and store the output in a hdf5 file """

    idx = int(batch_id)  # Read in the first argument

    #override with openmp settings
    if "OMP_NUM_THREADS" in os.environ.keys():
        openmp_ncores = int(os.environ["OMP_NUM_THREADS"])
        print(f"overriding nworkers {num_workers} -> {openmp_ncores}")
        num_workers = openmp_ncores

    #reproducibility
    np.random.seed(idx)
    torch.manual_seed(idx)

    # generating mass1 and ratio r=m2/m1
    # this way we ensure that mass2 is always less than mass1
    # as mass2 = r*mass1
    lower = torch.tensor([40.0, 0.25])
    upper = torch.tensor([80.0, 0.99])

    indep_uniform = torch.distributions.Uniform(lower, upper)
    #sample from the prior
    thetas = indep_uniform.sample((num_sims,)).view(-1, 2)

    print(
        f"simulating batch {idx} of {num_sims} simulated samples on {num_workers} detected cores"
    )
    value = sim_and_store(thetas, batch_id, batch_id, num_sims, num_workers, output, message)
    return value


# Run the simulator on multiple cores
# Note: This is neither a well written, nor a flexible script.
#       This is just there to run some simulations to demo sbi.
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)
