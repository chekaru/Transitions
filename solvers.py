import numpy as np
from scipy import integrate


class ReverseShootingSolver:

    @classmethod
    def solve(cls, model, t0, K0, dt, integrator, **solver_kwargs):

        # compute initial step size
        eps = 1e-12
        step = np.array([0, -eps]) if K0 <= model.equilibrium[1] else np.array([0, eps])
        initial_condition = (1 + step) * model.equilibrium

        # set up the integrator
        rhs = lambda t, X: -1 * np.array(model.rhs(t, X[0], X[1]))
        _ode = integrate.ode(rhs)
        _ode.set_integrator(integrator, **solver_kwargs)
        _ode.set_initial_value(initial_condition, t0)

        ts = np.array([t0])
        solution = initial_condition

        if K0 <= model.equilibrium[1]:

            while _ode.successful() and _ode.y[1] >= K0:
                ts = np.append(ts, _ode.t+dt)
                _ode.integrate(_ode.t+dt)
                solution = np.vstack((solution, _ode.y))
        else:

            while _ode.successful() and _ode.y[1] <= K0:
                ts = np.append(ts, _ode.t+dt)
                _ode.integrate(_ode.t+dt)
                solution = np.vstack((solution, _ode.y))

        # ordering is important!
        solution = np.hstack((ts[:, np.newaxis], solution[::-1, :]))
        solution = cls._compute_energy_prices(model, solution)
        solution = cls._compute_non_renewable_sector_output(model, solution)
        solution = cls._compute_renewable_sector_output(model, solution)
        solution = cls._compute_non_renewable_sector_costs(model, solution)
        solution = cls._compute_renewable_sector_costs(model, solution)
        solution = cls._compute_non_renewable_sector_profits(model, solution)
        solution = cls._compute_renewable_sector_profits(model, solution)

        return solution

    @staticmethod
    def _compute_energy_prices(model, solution):
        prices = np.array([model._compute_energy_price(K) for _, _, K in solution])
        return np.hstack((solution, prices[:, np.newaxis]))

    @staticmethod
    def _compute_energy_price_growth(ts, prices, deg=15):
        p_hat = np.polynomial.Chebyshev.fit(ts, prices, deg)
        return p_hat.deriv()(ts) / p_hat(ts)

    @staticmethod
    def _compute_non_renewable_sector_costs(model, solution):
        costs = np.empty((solution.shape[0], 1))
        for i, (q, capital, energy_price) in enumerate(solution[:, 1:4]):
            prices = (model._capital_price, energy_price, model._fossil_fuel_price)
            costs[i] = model._energy_market.non_renewable_sector.costs(q, capital, *prices)
        non_renewable_output = solution[:, [4]]
        return np.hstack((solution, costs / non_renewable_output))

    @classmethod
    def _compute_renewable_sector_costs(cls, model, solution):
        costs = np.empty((solution.shape[0], 1))
        ts, prices = solution[:, 0], solution[:, 3]
        growth_rates = cls._compute_energy_price_growth(ts, prices)
        for i, (price, growth_rate) in enumerate(zip(prices, growth_rates)):
            prices = (model._capital_price, price, growth_rate, model._interest_rate)
            costs[i] = model._energy_market.renewable_sector.costs(*prices)
        renewable_output = solution[:, [5]]
        return np.hstack((solution, costs / renewable_output))

    @staticmethod
    def _compute_non_renewable_sector_output(model, solution):
        energy = np.empty((solution.shape[0], 1))
        for i, (capital, energy_price) in enumerate(solution[:, 2:4]):
            prices = (energy_price, model._fossil_fuel_price)
            energy[i] = (model._energy_market.non_renewable_sector.output(capital, *prices))
        return np.hstack((solution, energy))

    @staticmethod
    def _compute_renewable_sector_output(model, solution):
        energy = np.empty((solution.shape[0], 1))
        energy_prices = solution[:, 3]
        for i, energy_price in enumerate(energy_prices):
            prices = (model._capital_price, energy_price, model._interest_rate)
            energy[i] = (model._energy_market.renewable_sector.output(*prices))
        return np.hstack((solution, energy))

    @staticmethod
    def _compute_non_renewable_sector_profits(model, solution):
        """Compute profits per unit energy output."""
        profits = np.empty((solution.shape[0], 1))
        for i, (q, capital, energy_price) in enumerate(solution[:, 1:4]):
            prices = (model._capital_price, energy_price, model._fossil_fuel_price)
            profits[i] = model._energy_market.non_renewable_sector.profits(q, capital, *prices)
        non_renewable_output = solution[:, [4]]
        return np.hstack((solution, profits / non_renewable_output))

    @classmethod
    def _compute_renewable_sector_profits(cls, model, solution):
        profits = np.empty((solution.shape[0], 1))
        ts, prices = solution[:, 0], solution[:, 3]
        growth_rates = cls._compute_energy_price_growth(ts, prices)
        for i, (price, growth_rate) in enumerate(zip(prices, growth_rates)):
            prices = (model._capital_price, price, growth_rate, model._interest_rate)
            profits[i] = model._energy_market.renewable_sector.profits(*prices)
        renewable_output = solution[:, [5]]
        return np.hstack((solution, profits / renewable_output))
