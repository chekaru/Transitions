import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


class TransitionDynamicsModel:

    def __init__(self, energy_market, capital_price, fossil_fuel_price, interest_rate):
        self._energy_market = energy_market

        self._capital_price = capital_price
        self._fossil_fuel_price = fossil_fuel_price
        self._interest_rate = interest_rate

    @property
    def equilibrium(self):
        """Equilibrium value for capital."""
        equilibrium_q = self._energy_market.non_renewable_sector.equilibrium_q
        equilibrium_capital = self.q_dot_locus(equilibrium_q)
        return np.array([equilibrium_q, equilibrium_capital])

    def plot_sector_costs(self, ts, qs, Ks, deg=35):

        # interpolate the energy price function...
        ps = np.array([self._compute_energy_price(K) for K in Ks])
        p_hat = np.polynomial.Chebyshev.fit(ts, ps, deg)
        grs = p_hat.deriv()(ts) / p_hat(ts)

        # compute the sector profits
        costs_NR = []
        costs_R = []
        for q, K, p, gr in zip(qs, Ks, ps, grs):
            costs_NR.append(self._compute_non_renewable_sector_costs(q, K, p))
            costs_R.append(self._compute_renewable_sector_costs(p, gr))

        fig, axes = plt.subplots(1, 2)

        axes[0].plot(ts, np.array(costs_NR), label=r"$C_{NR}(t)$")
        axes[0].legend(frameon=False)

        axes[1].plot(ts, np.array(costs_R), label=r"$C_{R}(t)$")
        axes[1].legend(frameon=False)

        fig.suptitle("Costs", fontsize=25, y=1.05, family='serif')
        fig.tight_layout()

        return fig

    def plot_sector_profits(self, ts, qs, Ks, deg=35):

        # interpolate the energy price function...
        ps = np.array([self._compute_energy_price(K) for K in Ks])
        p_hat = np.polynomial.Chebyshev.fit(ts, ps, deg)
        grs = p_hat.deriv()(ts) / p_hat(ts)

        # compute the sector profits
        profits_NR = []
        profits_R = []
        for q, K, p, gr in zip(qs, Ks, ps, grs):
            profits_NR.append(self._compute_non_renewable_sector_profits(q, K, p))
            profits_R.append(self._compute_renewable_sector_profits(p, gr))

        fig, axes = plt.subplots(1, 2)

        axes[0].plot(ts, np.array(profits_NR), label=r"$\Pi_{NR}(t)$")
        axes[0].legend(frameon=False)

        axes[1].plot(ts, np.array(profits_R), label=r"$\Pi_{R}(t)$")
        axes[1].legend(frameon=False)

        fig.suptitle("Profits", fontsize=25, y=1.05, family='serif')
        fig.tight_layout()

        return fig

    def plot_energy_price(self, ts, Ks):

        # compute energy prices
        energy_prices = ps = np.array([self._compute_energy_price(K) for K in Ks])

        fig, ax = plt.subplots(1, 1)

        ax.plot(ts, energy_prices, label=r"$p_E(t)$")
        ax.legend(frameon=False)

        ax.set_title("Energy Price", fontsize=25, family='serif')

        return fig

    def plot_sector_energy_output(self, ts, qs, Ks):

        # compute energy prices
        ps = np.array([self._compute_energy_price(K) for K in Ks])

        # compute the sector output
        output_NR = []
        output_R = []
        for K, p in zip(Ks, ps):
            output_NR.append(self._compute_non_renewable_sector_output(K, p))
            output_R.append(self._compute_renewable_sector_output(p))

        fig, axes = plt.subplots(1, 2)

        axes[0].plot(ts, np.array(output_NR) / self._energy_market.consumer.demand(ps), label=r"$E_{NR}(t)$")
        axes[0].set_ylim(0, 1)
        axes[0].legend(frameon=False)

        axes[1].plot(ts, np.array(output_R) / self._energy_market.consumer.demand(ps), label=r"$E_{R}(t)$")
        axes[1].set_ylim(0, 1)
        axes[1].legend(frameon=False)

        fig.suptitle("Energy output", fontsize=25, y=1.05, family='serif')
        fig.tight_layout()

        return fig

    def q_dot_locus(self, q):
        min_capital, max_capital = 1e-12, 1e6
        locus = lambda capital: self._q_dot(q, capital, self._compute_energy_price(capital))
        equilibrium_capital = optimize.brentq(locus, min_capital, max_capital)
        return equilibrium_capital

    def rhs(self, t, q, capital):
        energy_price = self._compute_energy_price(capital)
        return [self._q_dot(q, capital, energy_price), self._capital_dot(q, capital)]

    def _compute_energy_price(self, capital):
        """Can this be vectorized?"""
        prices = (self._capital_price, self._fossil_fuel_price, self._interest_rate)
        return self._energy_market.find_market_price(capital, *prices)

    def _capital_dot(self, q, capital):
        return self._energy_market.non_renewable_sector.equation_motion_capital(q, capital)

    def _q_dot(self, q, capital, energy_price):
        prices = (self._capital_price, energy_price, self._fossil_fuel_price, self._interest_rate)
        return self._energy_market.non_renewable_sector.equation_motion_q(q, capital, *prices)
