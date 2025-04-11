# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gravitational Wave Simulator with Bilby
#
# This notebook demonstrates how to generate gravitational wave signals for binary black holes (BBHs) using Bilby. It interfaces with the existing simulator structure (located in `./src/gw_simulator/simulator`) and allows you to switch between the PyCBC and Bilby implementations.
#
# Simply set the flag `SIMULATOR_TYPE` to either `'pycbc'` or `'bilby'` to choose your desired implementation.

# %%
import os
import sys

# Import bilby
# #!pip install bilby
import bilby
import matplotlib.pyplot as plt
import numpy as np
import torch
torch.random.manual_seed(43)
np.random.seed(42)

# %%
# Ensure the path to the simulator modules is set correctly
sys.path.insert(0, os.path.join(os.getcwd(), 'src/gw_simulator/simulator'))

# Import the existing PyCBC simulator and utility functions
from gw_simulator.simulator.interface import BilbyGravitationalWaveBenchmarkSimulator as gws
from gw_simulator.simulator.util import UniformMassPrior


# Example usage:
prior = UniformMassPrior(lower=10.0, upper=80.0)
sample = prior.sample((5,))  # Draw 5 samples
print("Sampled masses (mass1, mass2):\n", sample)
print("Log probability of samples:\n", prior.log_prob(sample))


# %%
masses = prior.sample((100_000,))
plt.scatter(masses[:, 0], masses[:, 1])
plt.xlabel("mass1")
plt.ylabel("mass2")
plt.savefig("gw_prior.svg")

# %%
# Instantiate the simulator based on the selected implementation
simulator = gws()

bilby.utils.logger.setLevel("ERROR")

# Draw a sample pair of masses from the prior
# (Using the custom UniformMassPrior since the original Prior had device issues)
prior = UniformMassPrior(lower=10.0, upper=80.0)
mass_samples = prior.sample((1,))  # sample one set (shape: (1, 2))

# Run the simulation
simulated_strains = simulator(mass_samples)

# Print shapes for debugging
for domain in simulated_strains.keys():
    print(f"DOMAIN: {domain}")
    for subdomain in simulated_strains[domain].keys():
        print(f"  {subdomain} shape: {simulated_strains[domain][subdomain].shape}")

# For the bilby simulator output, we now have a dictionary with keys 'detector' and 'noise'
# Each of those is a dict with keys 'time' and 'frequency'
masses = mass_samples[0]
detector_time_strains = simulated_strains["detector"]["time"][0].numpy()  # (N_ifos, T_td)
detector_freq_strains = simulated_strains["detector"]["frequency"][0].numpy()  # (N_ifos, T_fd)

noise_time_strains = simulated_strains["noise"]["time"][0].numpy()  # (N_ifos, T_td)
noise_freq_strains = simulated_strains["noise"]["frequency"][0].numpy()  # (N_ifos, T_fd)

# Plot time-domain strains for detector frame and noise
time_array = np.linspace(0, simulator.duration, detector_time_strains.shape[1])
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
for i, name in enumerate(simulator.ifo_names):
    plt.plot(time_array, detector_time_strains[i], label=name)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title(f'Detector Frame Time-Domain Strains (masses {masses})')
plt.legend()
#plt.savefig("bilby_det_frame_time_domain.svg")

plt.subplot(2, 1, 2)
for i, name in enumerate(simulator.ifo_names):
    plt.plot(time_array, noise_time_strains[i], label=name)
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Noise Time-Domain Strains')
plt.legend()

plt.tight_layout()
plt.savefig("gw_bilby_detnoise_time_domain.svg")
#plt.show()

# Plot frequency-domain strains for detector frame and noise
freq_bins = detector_freq_strains.shape[1]
frequency_array = np.linspace(0, simulator.sampling_frequency / 2, freq_bins)
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
for i, name in enumerate(simulator.ifo_names):
    plt.plot(frequency_array, detector_freq_strains[i], label=name)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Strain')
plt.title(f'Detector Frame Frequency-Domain Strains (masses {masses})')
plt.legend()

plt.subplot(2, 1, 2)
for i, name in enumerate(simulator.ifo_names):
    plt.plot(frequency_array, noise_freq_strains[i], label=name)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Strain')
plt.title('Noise Frequency-Domain Strains')
plt.legend()

plt.tight_layout()
#plt.show()
plt.savefig("gw_bilby_detnoise_frequ_domain.svg")
