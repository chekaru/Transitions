import numpy as np


def generate_consumer_params(seed=None):
    """Generate random parameters for a wholesale energy market consumer."""
    seed, prng = _generate_prng(seed)
    demand = prng.randint(1000000)
    params = {'quantity_demand': demand}
    return seed, params


def generate_non_renewable_sector_params(renewable_params, seed=None):
    """Generate random parameters for a RenewableEnergySector."""
    seed, prng = _generate_prng(seed)
    alpha = 1 - renewable_params['alpha']
    delta, phi, tfp = prng.lognormal(size=3)
    params = {'tfp': tfp, 'alpha': alpha, 'beta': 1 - alpha, 'delta': delta,
              'phi': phi, 'gamma': 1, 'sigma': 1}
    return seed, params


def generate_prices(seed=None):
    """Generate random capital price, fossil fuel price, and interest_rate."""
    seed, prng = _generate_prng(seed)
    capital_price, fossil_fuel_price, interest_rate = prng.lognormal(size=3)
    return seed, (capital_price, fossil_fuel_price, interest_rate)


def generate_renewable_sector_params(seed=None):
    """Generate random parameters for a RenewableEnergySector."""
    seed, prng = _generate_prng(seed)
    alpha = prng.rand()
    delta, mu, tfp = prng.lognormal(size=3)
    params = {'alpha': alpha, 'delta': delta, 'tfp': tfp, 'mu': mu}
    return seed, params


def _generate_prng(seed=None):
    """Generate seed for a random number generator."""
    if seed is None:
        high = np.iinfo(np.int32).max
        seed = np.random.randint(high)
    prng = np.random.RandomState(seed)
    return seed, prng
