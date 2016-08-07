import numpy as np


def _generate_non_renewable_sector_params(renewable_params, seed=None):
    """Generate random parameters for a RenewableEnergySector."""
    prng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    alpha = 1 - renewable_params['alpha']
    delta, phi, tfp = prng.lognormal(size=3)
    params = {'tfp': tfp, 'alpha': alpha, 'beta': 1 - alpha, 'delta': delta,
              'phi': phi, 'gamma': 1, 'sigma': 1}
    return params


def _generate_renewable_sector_params(seed=None):
    """Generate random parameters for a RenewableEnergySector."""
    prng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    alpha = prng.rand()
    delta, mu, tfp = prng.lognormal(size=3)
    params = {'alpha': alpha, 'delta': delta, 'tfp': tfp, 'mu': mu}
    return params
