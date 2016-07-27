from scipy import optimize


class RenewableEnergySector:

    @staticmethod
    def output(capital, params):
        """Renewable energy output."""
        energy = params['tfp'] * capital**params['alpha']
        assert energy > 0, "Renewable energy output of {} is not positive!".format(energy)
        return energy

    @classmethod
    def profits(cls, capital, energy_price, params):
        """Renewable energy sector profits."""
        pi = (cls.revenue(capital, energy_price, params) -
              cls.total_costs(capital, params))
        return pi

    @classmethod
    def revenue(cls, capital, energy_price, params):
        """Renewable energy revenue."""
        return cls.subsidy(energy_price, params) * cls.output(capital, params)

    @staticmethod
    def subsidy(energy_price, params):
        """For now assume that subsidy is the energy price."""
        return energy_price

    @staticmethod
    def total_costs(capital, params):
        """Renewable energy production costs."""
        return params['gross_interest_rate'] * capital

    @classmethod
    def capital_demand(cls, energy_price, params):
        """
        Renewable energy sector demand for physical capital.

        Note
        ----
        The demand for physical capital is a function of the price of energy
        (which may or may not be subsidized) and model parameters.

        """
        relative_price = cls.subsidy(energy_price, params) / params['gross_interest_rate']
        demand = (params['alpha'] * params['tfp'] * relative_price**(1 - params['alpha']))
        assert demand > 0, "Renewable energy sector capital demand of {} is not positive!".format(demand)
        return demand


class NonRenewableEnergySector:

    @classmethod
    def equation_motion_capital(cls, t, q, capital, delta, **params):
        """Differential equation describing the time evolution of capital."""
        I = cls._investment_demand(q, capital, **params)
        K_dot = I - delta * capital
        return K_dot

    @classmethod
    def equation_motion_q(cls, t, q, capital, capital_price, energy_price, fossil_fuel_price, r, delta, **params):
        """Differential equation describing the time evolution of Tobin's q."""
        fossil_fuel = cls._fossil_fuel_demand(capital, energy_price, fossil_fuel_price, **params)
        investment = cls._investment_demand(q, capital, **params)
        q_dot = ((r + delta) * q +
                 cls._marginal_percentage_adjustment_costs(capital, investment, **params) * investment -
                 cls._value_marginal_product_capital(capital, fossil_fuel, energy_price, **params) / capital_price)
        return q_dot

    @classmethod
    def output(cls, capital, fossil_fuel, tfp, alpha, beta, gamma, sigma, **params):
        """
        Non-renewable sector energy output.

        Notes
        -----
        Sector uses a generalized version of the constant elasticity of
        substitution (CES) functional form.

        """
        if cls._is_cobb_douglas(alpha, beta, gamma, sigma):
            energy = tfp * capital**alpha * fossil_fuel**beta
        elif (rho == 0):
            raise NotImplementedError
        else:
            rho = (sigma - 1) / sigma
            energy = tfp * (alpha * capital**rho + beta * fossil_fuel**rho)**(gamma / rho)
        return energy

    @classmethod
    def profits(cls, capital, fossil_fuel, investment, capital_price, energy_price, fossil_fuel_price, **params):
        """Non-renewable sector profits."""
        pi = (cls._revenue(capital, fossil_fuel, energy_price, **params) -
              cls._production_costs(capital, fossil_fuel, investment, capital_price, fossil_fuel_price, **params))
        return pi

    @classmethod
    def _cost_capital(cls, capital, investment, capital_price, **params):
        """Total costs of capital for use in producing energy from fossil fuels."""
        costs = capital_price * (1 + cls._percentage_adjustment_costs(capital, investment, **params)) * investment
        return costs

    @staticmethod
    def _cost_fossil_fuel(fossil_fuel, fossil_fuel_price, **params):
        """Non-renewable sector production costs from fossil fuels."""
        costs = fossil_fuel_price * fossil_fuel
        return costs

    @staticmethod
    def _is_cobb_douglas(alpha, beta, gamma, sigma, **params):
        """Check whether parameters imply Cobb-Douglas functional form."""
        rho = (sigma - 1) / sigma
        return (rho == 0) and ((alpha + beta) == gamma)

    @staticmethod
    def _investment_demand(q, capital, phi, **params):
        """Non-renewable energy sector demand for investment."""
        demand = ((2 / 3) * (q - 1) * (1 / phi))**0.5 * capital
        return demand

    @classmethod
    def _fossil_fuel_demand(cls, capital, energy_price, fossil_fuel_price, tfp, alpha, beta, **params):
        """Non-renewable energy sector demand for fossil fuels."""
        relative_price = fossil_fuel_price / energy_price
        if cls._is_cobb_douglas(alpha, beta, **params):
            demand = (tfp * beta / relative_price)**(1 / (1 - beta)) * capital**(alpha / (1 - beta))
        else:
            raise NotImplementedError
        return demand

    @classmethod
    def _marginal_percentage_adjustment_costs(cls, capital, investment, phi, **params):
        """Non-renewable sector marginal costs of capital."""
        mc = -phi * (investment / capital)**2 * (1 / capital)
        return mc

    @classmethod
    def _marginal_product_capital(cls, capital, fossil_fuel, tfp, alpha, beta,
                                  gamma, sigma, **params):
        """Non-renewable sector marginal product of capital."""
        if cls._is_cobb_douglas(alpha, beta, gamma, sigma):
            mpk = tfp * alpha * capital**(alpha - 1) * fossil_fuel**beta
        elif (rho == 0):
            raise NotImplementedError
        else:
            rho = (sigma - 1) / sigma
            mpk = tfp * alpha * gamma * capital**(rho - 1) * (alpha * capital**rho + beta * fossil_fuel**rho)**((gamma / rho) - 1)
        return mpk

    @classmethod
    def _marginal_product_fossil_fuel(cls, capital, fossil_fuel, tfp, alpha, beta,
                                      gamma, sigma, **params):
        """Non-renewable sector marginal product of capital."""
        if cls._is_cobb_douglas(alpha, beta, gamma, sigma):
            mpf = tfp * beta * capital**alpha * fossil_fuel**(beta - 1)
        elif (rho == 0):
            raise NotImplementedError
        else:
            rho = (sigma - 1) / sigma
            mpf = tfp * beta * gamma * fossil_fuel**(rho - 1) * (alpha * capital**rho + beta * fossil_fuel**rho)**((gamma / rho) - 1)
        return mpf

    @staticmethod
    def _percentage_adjustment_costs(capital, investment, phi, **params):
        """Convex capital adjustment cost function."""
        costs = (phi / 2) * (investment / capital)**2
        return costs

    @classmethod
    def _production_costs(cls, capital, previous_capital, fossil_fuel, **params):
        """Non-renewable sector production costs."""
        costs = (cls._cost_capital(capital, previous_capital, **params) +
                 cls._cost_fossil_fuel(fossil_fuel, **params))
        return costs

    @classmethod
    def _revenue(cls, capital, fossil_fuel, energy_price, **params):
        """Non-renewable sector revenue."""
        return energy_price * cls.output(capital, fossil_fuel, **params)

    @classmethod
    def _value_marginal_product_capital(cls, capital, fossil_fuel, energy_price, **params):
        """Non-renewable sector value marginal product of capital."""
        vmp = energy_price * cls._marginal_product_capital(capital, fossil_fuel, **params)
        return vmp
