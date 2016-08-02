from scipy import optimize


class RenewableEnergySector:

    def __init__(self, tfp, alpha, delta, mu):
        self._tfp = tfp
        self._alpha = alpha
        self._delta = delta
        self._mu = mu

    def output(self, capital_price, energy_price, interest_rate):
        """Renewable energy sector output."""
        capital = self._capital_demand(capital_price, energy_price, interest_rate)
        energy = self._tfp * capital**self._alpha
        return energy

    def profits(self, capital_price, energy_price, energy_price_growth, interest_rate):
        """Renewable energy sector profits."""
        pi = (self._revenue(capital_price, energy_price, interest_rate) -
              self.costs(capital_price, energy_price, energy_price_growth, interest_rate))
        return pi

    def subsidy(self, energy_price):
        """Subsidized price of renewable energy."""
        return (1 + self._mu) * energy_price

    def _capital_demand(self, capital_price, energy_price, interest_rate):
        """Renewable energy sector demand for capital."""
        relative_price = capital_price / self.subsidy(energy_price)
        demand = ((self._alpha * self._tfp / (interest_rate + self._delta)) * (1 / relative_price))**(1 / (1 - self._alpha))
        return demand

    def costs(self, capital_price, energy_price, energy_price_growth, interest_rate):
        """Renewable energy production costs."""
        capital = self._capital_demand(capital_price, energy_price, interest_rate)
        return ((1 / (1 - self._alpha)) * (energy_price_growth) + self._delta) * capital

    def _revenue(self, capital_price, energy_price, interest_rate):
        """Renewable energy revenue."""
        return self.subsidy(energy_price) * self.output(capital_price, energy_price, interest_rate)


class NonRenewableEnergySector:

    def __init__(self, tfp, alpha, beta, gamma, delta, phi, sigma):
        self._tfp = tfp
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._phi = phi
        self._rho = (sigma - 1) / sigma
        self._sigma = sigma

    def equation_motion_capital(self, q, capital):
        """Differential equation describing the time evolution of capital."""
        K_dot = self._investment_demand(q, capital) - self._delta * capital
        return K_dot

    def equation_motion_q(self, q, capital, capital_price, energy_price, fossil_fuel_price, interest_rate):
        """Differential equation describing the time evolution of Tobin's q."""
        I = self._investment_demand(q, capital)
        q_dot = ((interest_rate + self._delta) * q +
                 self._marginal_percentage_adjustment_costs(capital, I) * I -
                 self._value_marginal_product_capital(capital, energy_price, fossil_fuel_price) / capital_price)
        return q_dot

    @property
    def equilibrium_q(self):
        """Equilibrium value for Tobin's q."""
        return 1 + (3 / 2) * self._delta**2 * self._phi

    def output(self, capital, energy_price, fossil_fuel_price):
        """Non-renewable sector energy output."""
        F = self._fossil_fuel_demand(capital, energy_price, fossil_fuel_price)
        if self._is_cobb_douglas:
            energy = self._tfp * capital**self._alpha * F**self._beta
        elif (self._rho == 0):
            energy = self._tfp * (capital**self._alpha * F**self._beta)**self._gamma
        else:
            energy = self._tfp * (self._alpha * capital**self._rho + self._beta * F**self._rho)**(self._gamma / self._rho)
        return energy

    def profits(self, q, capital, capital_price, energy_price, fossil_fuel_price):
        """Non-renewable sector profits."""
        pi = (self._revenue(capital, energy_price, fossil_fuel_price) -
              self.costs(q, capital, capital_price, energy_price, fossil_fuel_price))
        return pi

    def costs(self, q, capital, capital_price, energy_price, fossil_fuel_price):
        """Non-renewable sector production costs."""
        costs = (self._cost_capital(q, capital, capital_price) +
                 self._cost_fossil_fuel(capital, energy_price, fossil_fuel_price))
        return costs

    def _cost_capital(self, q, capital, capital_price):
        """Total costs of capital for use in producing energy from fossil fuels."""
        I = self._investment_demand(q, capital)
        costs = capital_price * (1 + self._percentage_adjustment_costs(capital, I)) * I
        return costs

    def _cost_fossil_fuel(self, capital, energy_price, fossil_fuel_price):
        """Non-renewable sector production costs from fossil fuels."""
        F = self._fossil_fuel_demand(capital, energy_price, fossil_fuel_price)
        costs = fossil_fuel_price * F
        return costs

    def _fossil_fuel_demand(self, capital, energy_price, fossil_fuel_price):
        """Non-renewable energy sector demand for fossil fuels."""
        relative_price = fossil_fuel_price / energy_price
        if self._is_cobb_douglas:
            demand = (self._tfp * self._beta / relative_price)**(1 / (1 - self._beta)) * capital**(self._alpha / (1 - self._beta))
        else:
            raise NotImplementedError  # need to use fsolve on FOC!
        return demand

    @property
    def _is_cobb_douglas(self):
        """Check whether parameters imply Cobb-Douglas functional form."""
        return (self._rho == 0) and ((self._alpha + self._beta) == self._gamma)

    def _investment_demand(self, q, capital):
        """Non-renewable energy sector demand for investment."""
        demand = ((2 / 3) * (q - 1) * (1 / self._phi))**0.5 * capital
        return demand

    def _marginal_percentage_adjustment_costs(self, capital, investment):
        """Non-renewable sector marginal costs of capital."""
        mc = -self._phi * (investment / capital)**2 * (1 / capital)
        return mc

    def _marginal_product_capital(self, capital, energy_price, fossil_fuel_price):
        """Non-renewable sector marginal product of capital."""
        F = self._fossil_fuel_demand(capital, energy_price, fossil_fuel_price)
        energy = self.output(capital, energy_price, fossil_fuel_price)
        if self._is_cobb_douglas:
            mpk = self._alpha * (energy / capital)
        elif (self._rho == 0):
            mpk = self._alpha * self._gamma * (energy / capital)
        else:
            capital_share = (self._alpha * self._gamma * capital**self._rho /
                             (self._alpha * capital**self._rho + self._beta * F**rho))
            mpk = capital_share * (energy / capital)
        return mpk

    def _marginal_product_fossil_fuel(self, capital, energy_price, fossil_fuel_price):
        """Non-renewable sector marginal product of fossil fuels."""
        F = self._fossil_fuel_demand(capital, energy_price, fossil_fuel_price)
        energy = self.output(capital, energy_price, fossil_fuel_price)
        if self._is_cobb_douglas:
            mpf = self._beta * (energy / F)
        elif (self._rho == 0):
            mpf = self._beta * self._gamma * (energy / F)
        else:
            fossil_fuel_share = ((self._beta * self._gamma * F**self._rho) /
                                 (self._alpha * capital**self._rho + self._beta * F**self._rho))
            mpf = fossil_fuel_share * (energy / fossil_fuel)
        return mpf

    def _percentage_adjustment_costs(self, capital, investment):
        """Convex capital adjustment cost function."""
        costs = (self._phi / 2) * (investment / capital)**2
        return costs

    def _revenue(self, capital, energy_price, fossil_fuel_price):
        """Non-renewable sector revenue."""
        return energy_price * self.output(capital, energy_price, fossil_fuel_price)

    def _value_marginal_product_capital(self, capital, energy_price, fossil_fuel_price):
        """Non-renewable sector value marginal product of capital."""
        vmp = energy_price * self._marginal_product_capital(capital, energy_price, fossil_fuel_price)
        return vmp
