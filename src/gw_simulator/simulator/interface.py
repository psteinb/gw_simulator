r"""Simulator definition of the Gravitational Wave benchmark problem.

This specific model marginalizes over the mass parameters of the black holes.
The problem dimensionality of the inputs therefore reduces to 2.

Inspired by https://github.com/timothygebhard/ggwd
"""

import bilby
import torch


class BaseSimulator:
    r"""Base simulator class.

    A simulator defines the implicit forward model.

    Example usage of a potential simulator implementation:

        simulator = MySimulator()
        inputs = prior.sample((10,)) # Draw 10 samples from the prior.
        outputs = simulator(inputs)

    In principle, this corresponds to sampling from the joint $$\vartheta,x\sim p(\vartheta)p(x\vert\vartheta)$$,
    where $$p(x\vert\vartheta)$$ is the likelihood-model implicitely defined through the simulator.

    .. note::

        The ``inputs`` and ``outputs`` variable name in most simulator denote
        their position with respect to the simulation model. ``inputs`` are
        typically free parameters of the simulation model which sample
        (or produce deterministically) ``outputs``.

    .. note::

        Although it is possibly to supply a batch of inputs, it should be
        noted that these are currently `not` parallelized.

    """

    def __init__(self):
        super(BaseSimulator, self).__init__()

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def forward(self, inputs, **kwargs):
        r"""Defines the computation of the forward model at every call.

        .. note::

            Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __del__(self):
        self.terminate()

    def terminate(self):
        r"""Terminates the simulator and cleans up possible contexts.

        .. note::

            Should be overridden by subclasses with a simulator state requiring graceful exits.
        """
        pass


class BilbyGravitationalWaveBenchmarkSimulator(BaseSimulator):
    """
    Simulator model for gravitational waves using Bilby.
    Returns separate outputs for:
      - detector frame strains (i.e. the injected gravitational wave signal in the detector, with zero noise)
      - noise strains (the noise realization used)
    Both are provided in time and frequency domains.
    """
    def __init__(self,
                 duration=4.0,
                 sampling_frequency=2048,
                 ifo_names=['H1', 'L1', 'V1'],
                 waveform_approximant='IMRPhenomPv2'):
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.ifo_names = ifo_names
        self.waveform_approximant = waveform_approximant

        # Set up default waveform arguments (adjust as needed)
        self.waveform_arguments = dict(
            waveform_approximant=self.waveform_approximant,
            reference_frequency=20.0,
            minimum_frequency=20.0
        )

        # Create a Bilby waveform generator using the LAL binary black hole source model
        self.waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration,
            sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=self.waveform_arguments
        )

        # Default geocentric time for injections.
        self.merger_time = 1126259462.4
        # Offset so that the merger is not at the very beginning of the time series.
        self.start_offset = 3.0

    def _simulate_gw(self, mass1, mass2):
        # Set up injection parameters (modify as needed)
        parameters = {
            'mass_1': float(mass1),
            'mass_2': float(mass2),
            'luminosity_distance': 400.0,  # in Mpc
            'theta_jn': 0.4,              # inclination angle (radians)
            'phase': 0.0,
            'geocent_time': self.merger_time,
            'ra': 1.95,
            'dec': -1.2,
            'psi': 0.0,
            'a_1': 0.0,
            'a_2': 0.0,
            'tilt_1': 0.0,
            'tilt_2': 0.0,
        }

        # Create interferometers.
        ifos = bilby.gw.detector.InterferometerList(self.ifo_names)

        # First, set the strain data from the PSD to generate a noise realization.
        # Store the noise separately for each interferometer.
        noise_data = {}
        for ifo in ifos:
            ifo.set_strain_data_from_power_spectral_density(
                sampling_frequency=self.sampling_frequency,
                duration=self.duration,
                start_time=self.merger_time - self.start_offset
            )
            # Copy the noise realization in both time and frequency domains.
            noise_td = ifo.strain_data.time_domain_strain.copy()
            noise_fd = ifo.strain_data.frequency_domain_strain.copy()
            noise_data[ifo.name] = {"td": noise_td, "fd": noise_fd}

        # Now inject the gravitational wave signal into the interferometers.
        ifos.inject_signal(parameters=parameters, waveform_generator=self.waveform_generator)

        # Retrieve the full (signal + noise) data and compute the pure signal (detector frame strain).
        strains = {}
        for ifo in ifos:
            full_td = ifo.strain_data.time_domain_strain
            full_fd = ifo.strain_data.frequency_domain_strain
            # Compute the pure injected signal by subtracting the noise realization.
            signal_td = full_td - noise_data[ifo.name]["td"]
            signal_fd = full_fd - noise_data[ifo.name]["fd"]
            strains[ifo.name] = {
                "detector": {"td": signal_td, "fd": signal_fd},
                "noise": {"td": noise_data[ifo.name]["td"], "fd": noise_data[ifo.name]["fd"]},
                "full": {"td": full_td, "fd": full_fd}  # full = detector + noise
            }
        return strains

    def forward(self, inputs, **kwargs):
        """
        Simulate gravitational wave signals for a batch of mass parameters.

        Args:
            inputs (torch.Tensor): A tensor of shape (batch_size, 2) containing mass parameters.

        Returns:
            dict: A dictionary with two keys ('detector' and 'noise'), each mapping to a sub-dictionary:
                  - "time": tensor of shape (batch_size, N_ifos, T_td)
                  - "frequency": tensor of shape (batch_size, N_ifos, T_fd)
        """
        batch_detector_td = []
        batch_detector_fd = []
        batch_noise_td = []
        batch_noise_fd = []

        # Loop over each sample in the batch.
        for mass_pair in inputs.view(-1, 2):
            strains = self._simulate_gw(mass_pair[0].item(), mass_pair[1].item())
            detector_td_list = []
            detector_fd_list = []
            noise_td_list = []
            noise_fd_list = []
            # Ensure ordering of interferometers follows self.ifo_names.
            for name in self.ifo_names:
                detector_td_list.append(torch.tensor(strains[name]["detector"]["td"]))
                detector_fd_list.append(torch.tensor(strains[name]["detector"]["fd"]))
                noise_td_list.append(torch.tensor(strains[name]["noise"]["td"]))
                noise_fd_list.append(torch.tensor(strains[name]["noise"]["fd"]))
            batch_detector_td.append(torch.stack(detector_td_list, dim=0))
            batch_detector_fd.append(torch.stack(detector_fd_list, dim=0))
            batch_noise_td.append(torch.stack(noise_td_list, dim=0))
            batch_noise_fd.append(torch.stack(noise_fd_list, dim=0))

        return {
            "detector": {
                "time": torch.stack(batch_detector_td, dim=0),
                "frequency": torch.stack(batch_detector_fd, dim=0)
            },
            "noise": {
                "time": torch.stack(batch_noise_td, dim=0),
                "frequency": torch.stack(batch_noise_fd, dim=0)
            }
        }

    def terminate(self):
        # Cleanup if needed.
        pass
