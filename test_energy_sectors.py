import json

import numpy as np

from energy_sectors import NonRenewableEnergySector


def _generate_cobb_douglas_params(seed=None, crts=False):
    """Generates parameters consistent with Cobb-Douglas functional form."""
    prng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    alpha = prng.rand()
    if crts:
        params =  {'alpha': alpha, 'beta': 1 - alpha, 'gamma': 1, 'sigma': 1}
    else:
        beta = prng.rand()
        params =  {'alpha': alpha, 'beta': beta, 'gamma': alpha + beta, 'sigma': 1}
    return params


def _generate_ces_params(seed=None, crts=False):
    """Generates parameters consistent with CES functional form."""
    prng = np.random.RandomState() if seed is None else np.random.RandomState(seed)
    alpha = prng.rand()
    beta = prng.rand()
    sigma = prng.lognormal()
    if crts:
        params =  {'alpha': alpha, 'beta': beta, 'gamma': 1, 'sigma': sigma}
    else:
        gamma = prng.lognormal()
        params =  {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'sigma': sigma}
    return params


def test_output():

    # zero capital should produce zero output!
    params = _generate_ces_params()
    mesg = json.dumps(params)
    assert NonRenewableEnergySector.output(0, 10, **params) == 0, mesg

    params = _generate_cobb_douglas_params()
    mesg = json.dumps(params)
    assert NonRenewableEnergySector.output(0, 10, **params) == 0

    # zero fossil fuels should produce zero output!
    params = _generate_ces_params()
    mesg = json.dumps(params)
    assert NonRenewableEnergySector.output(100, 0, **params) == 0

    params = _generate_cobb_douglas_params()
    mesg = json.dumps(params)
    assert NonRenewableEnergySector.output(100, 0, **params) == 0

    # zero inputs should produce zero output!
    params = _generate_ces_params()
    mesg = json.dumps(params)
    assert NonRenewableEnergySector.output(0, 0, **params) == 0

    params = _generate_cobb_douglas_params()
    mesg = json.dumps(params)
    assert NonRenewableEnergySector.output(0, 0, **params) == 0
