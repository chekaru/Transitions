import numpy as np
from scipy import integrate


class ReverseShootingSolver:

    def __init__(self, model):
        self._model = model

    def solve(self, t0, K0, dt, integrator, **solver_kwargs):

        # compute initial step size
        eps = 1e-12
        step = np.array([0, -eps]) if K0 <= self._model.equilibrium[1] else np.array([0, eps])
        initial_condition = (1 + step) * self._model.equilibrium

        # set up the integrator
        _ode = integrate.ode(self._rhs)
        _ode.set_integrator(integrator, **solver_kwargs)
        _ode.set_initial_value(initial_condition, t0)

        ts = np.array([t0])
        solution = initial_condition

        if K0 <= self._model.equilibrium[1]:

            while _ode.successful() and _ode.y[1] >= K0:
                ts = np.append(ts, _ode.t+dt)
                _ode.integrate(_ode.t+dt)
                solution = np.vstack((solution, _ode.y))
        else:

            while _ode.successful() and _ode.y[1] <= K0:
                ts = np.append(ts, _ode.t+dt)
                _ode.integrate(_ode.t+dt)
                solution = np.vstack((solution, _ode.y))

        trajectory = np.hstack((ts[:, np.newaxis], solution[::-1, :]))
        trajectory = self._compute_energy_prices(trajectory)
        trajectory = self._compute_non_renewable_sector_output(trajectory)
        trajectory = self._compute_renewable_sector_output(trajectory)
        trajectory = self._compute_non_renewable_sector_costs(trajectory)
        trajectory = self._compute_renewable_sector_costs(trajectory)
        trajectory = self._compute_non_renewable_sector_profits(trajectory)
        trajectory = self._compute_renewable_sector_profits(trajectory)

        return trajectory

    def _compute_energy_prices(self, trajectory):
        prices = np.array([self._model._compute_energy_price(K) for _, _, K in trajectory])
        return np.hstack((trajectory, prices[:, np.newaxis]))

    @staticmethod
    def _compute_energy_price_growth(ts, prices, deg=15):
        p_hat = np.polynomial.Chebyshev.fit(ts, prices, deg)
        return p_hat.deriv()(ts) / p_hat(ts)

    def _compute_non_renewable_sector_costs(self, trajectory):
        costs = np.empty((trajectory.shape[0], 1))
        for i, (q, capital, energy_price) in enumerate(trajectory[:, 1:4]):
            prices = (self._model._capital_price, energy_price, self._model._fossil_fuel_price)
            costs[i] = self._model._energy_market.non_renewable_sector.costs(q, capital, *prices)
        return np.hstack((trajectory, costs))

    def _compute_renewable_sector_costs(self, trajectory):
        costs = np.empty((trajectory.shape[0], 1))
        ts, prices = trajectory[:, 0], trajectory[:, 3]
        growth_rates = self._compute_energy_price_growth(ts, prices)
        for i, (price, growth_rate) in enumerate(zip(prices, growth_rates)):
            prices = (self._model._capital_price, price, growth_rate, self._model._interest_rate)
            costs[i] = self._model._energy_market.renewable_sector.costs(*prices)
        return np.hstack((trajectory, costs))

    def _compute_non_renewable_sector_output(self, trajectory):
        energy = np.empty((trajectory.shape[0], 1))
        for i, (capital, energy_price) in enumerate(trajectory[:, 2:4]):
            prices = (energy_price, self._model._fossil_fuel_price)
            energy[i] = (self._model._energy_market.non_renewable_sector.output(capital, *prices))
        return np.hstack((trajectory, energy))

    def _compute_renewable_sector_output(self, trajectory):
        energy = np.empty((trajectory.shape[0], 1))
        energy_prices = trajectory[:, 3]
        for i, energy_price in enumerate(energy_prices):
            prices = (self._model._capital_price, energy_price, self._model._interest_rate)
            energy[i] = (self._model._energy_market.renewable_sector.output(*prices))
        return np.hstack((trajectory, energy))

    def _compute_non_renewable_sector_profits(self, trajectory):
        profits = np.empty((trajectory.shape[0], 1))
        for i, (q, capital, energy_price) in enumerate(trajectory[:, 1:4]):
            prices = (self._model._capital_price, energy_price, self._model._fossil_fuel_price)
            profits[i] = self._model._energy_market.non_renewable_sector.profits(q, capital, *prices)
        return np.hstack((trajectory, profits))

    def _compute_renewable_sector_profits(self, trajectory):
        profits = np.empty((trajectory.shape[0], 1))
        ts, prices = trajectory[:, 0], trajectory[:, 3]
        growth_rates = self._compute_energy_price_growth(ts, prices)
        for i, (price, growth_rate) in enumerate(zip(prices, growth_rates)):
            prices = (self._model._capital_price, price, growth_rate, self._model._interest_rate)
            profits[i] = self._model._energy_market.renewable_sector.profits(*prices)
        return np.hstack((trajectory, profits))

    def _rhs(self, t, X):
        """Wrapper for the RHS of the model so it confirms with integrate.ode API."""
        gradient = np.array(self._model.rhs(t, X[0], X[1]))
        return -1 * gradient  # don't forget to reverse time!
