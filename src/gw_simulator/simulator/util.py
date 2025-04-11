r"""Utilities for the Gravitational Wave benchmark.

"""

import torch


class UniformMassPrior:
    """
    Uniform prior on two masses with the constraint mass1 > mass2.

    The allowed region is:
      lower <= mass2 < mass1 <= upper

    The sampling procedure generates two independent uniform samples in [lower, upper]
    and then sorts them so that mass1 is the maximum (first element) and mass2 the minimum (second element).

    The log probability is constant (2/(upper - lower)²) over the allowed region.
    """
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
        # The constant density for a uniform distribution over the triangle of area (upper-lower)²/2.
        self.log_density = torch.log(torch.tensor(2.0 / ((upper - lower) ** 2)))

    def sample(self, sample_shape=torch.Size((1,))):
        """
        Draw samples from the constrained uniform prior.

        Args:
            sample_shape (torch.Size or tuple): Shape of samples to generate (default is one sample).

        Returns:
            torch.Tensor: A tensor of shape (*sample_shape, 2) with mass1 and mass2.
        """
        # Draw two independent samples in [0,1]
        samples = torch.rand(*sample_shape, 2)
        # Scale them to [lower, upper]
        samples = self.lower + (self.upper - self.lower) * samples
        # Sort along the last dimension in ascending order
        sorted_samples, _ = torch.sort(samples, dim=-1)
        # Rearrange so that mass1 (largest) is first and mass2 (smallest) is second
        mass1 = sorted_samples[..., 1]
        mass2 = sorted_samples[..., 0]
        return torch.stack([mass1, mass2], dim=-1)

    def log_prob(self, sample: torch.Tensor):
        """
        Evaluate the log-probability of a given sample.

        Args:
            sample (torch.Tensor): A tensor of shape (..., 2) where the first column is mass1 and second mass2.

        Returns:
            torch.Tensor: Log probability (scalar or tensor) for the sample. Returns -inf if constraints are violated.
        """
        m1 = sample[..., 0]
        m2 = sample[..., 1]
        # Check bounds and constraint: mass1 > mass2
        valid = (m1 >= self.lower) & (m1 <= self.upper) & (m2 >= self.lower) & (m2 <= self.upper) & (m1 > m2)
        # Assign constant log density if valid; otherwise, return -infinity
        log_p = torch.where(valid, self.log_density, torch.tensor(float('-inf')))
        return log_p
