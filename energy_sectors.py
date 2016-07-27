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
    def output(cls, capital, fossil_fuel, tfp, alpha, beta, gamma, sigma, **params):
        """
        Non-renewable sector energy output.

        Notes
        -----
        Sector uses a generalized version of the constant elasticity of
        substitution (CES) functional form.

        """
        if cls._is_cobb_douglas(alpha, beta, gamma, sigma):
            energy = tfp * K**alpha * F**beta
        else:
            energy = tfp * (alpha * K**rho + beta * F**rho)**(gamma / rho)
        assert energy > 0, "Non-renewable energy output is not positive!"
        return energy

    @classmethod
    def profits(cls, capital, fossil_fuel, investment, capital_price, energy_price, fossil_fuel_price, **params):
        """Non-renewable sector profits."""
        pi = (cls._revenue(capital, fossil_fuel, energy_price, **params) -
              cls._production_costs(capital, fossil_fuel, investment, capital_price, fossil_fuel_price, **params))
        return pi

    @classmethod
    def marginal_cost_capital(cls, capital, previous_capital, params):
        """Costs of adjusting stock of physical capital."""
        mc = (params['gross_interest_rate'] +
              cls.marginal_capital_adjustment_costs(capital, previous_capital, params))
        return mc

    @staticmethod
    def marginal_capital_adjustment_costs(capital, previous_capital, params):
        """Marginal costs of adjusting stock of capital."""
        return params['phi'] * (capital - previous_capital)

    @classmethod
    def capital_output_ratio(cls, capital, fossil_fuel, params):
        """Ratio of capital stock to output."""
        ratio =  capital / cls.output(capital, fossil_fuel, params)
        assert ratio > 0, "Non-renewable energy sector capital-output ratio of {} is not positive!".format(ratio)
        return ratio

    @classmethod
    def value_marginal_product_capital(cls, capital, fossil_fuel, energy_price, params):
        """Contribution to firm revenue of the marginal unit of installed capital."""
        vmp = energy_price * cls.marginal_product_capital(capital, fossil_fuel, energy_price, params)
        assert vmp > 0, "Non-renewable energy sector value marginal product of capital of {} is not positive!".format(vmp)
        return vmp

    @classmethod
    def net_value_marginal_product_capital(cls, capital, previous_capital, fossil_fuel, energy_price, params):
        """Value marginal product of capital less marginal costs of capital."""
        nvmp = (cls.value_marginal_product_capital(capital, fossil_fuel, energy_price, params) -
                cls.marginal_cost_capital(capital, previous_capital, params))
        return nvmp

    @classmethod
    def future_capital_demand(cls, capital, previous_capital, fossil_fuel, energy_price, params):
        """Demand for capital is forward looking?"""
        nvmp = cls.net_value_marginal_product_capital(capital, previous_capital, fossil_fuel, energy_price, params)
        demand = capital + (nvmp / (params['eta'] * params['phi']))
        assert demand > 0, "Non-renewable energy sector capital demand of {} is not positive!".format(demand)
        return demand

    @staticmethod
    def fossil_fuel_demand(capital, energy_price, params):
        """Non-renewable energy sector demand for fossil fuels."""
        relative_price = energy_price / params['fossil_fuel_price']
        demand = (params['tfp'] * (1 - params['beta']) * relative_price)**(1 / params['beta']) * capital
        assert demand > 0, "Non-renewable energy sector fossil fuel demand of {} is not positive!".format(demand)
        return demand

    @classmethod
    def _cost_capital(cls, capital, investment, capital_price, **params):
        """Total costs of capital for use in producing energy from fossil fuels."""
        costs = capital_price * (1 + cls._percentage_adjustment_costs(capital, investment, **params)) * investment
        assert costs > 0, "Non-renewable energy sector capital costs are not positive!"
        return costs

    @staticmethod
    def _cost_fossil_fuel(fossil_fuel, fossil_fuel_price, **params):
        """Non-renewable sector production costs from fossil fuels."""
        costs = fossil_fuel_price * fossil_fuel
        assert costs > 0, "Non-renewable sector fossil fuel costs are not positive!"
        return costs

    @staticmethod
    def _is_cobb_douglas(alpha, beta, gamma, sigma, **params):
        """Check whether parameters imply Cobb-Douglas functional form."""
        rho = (sigma - 1) / sigma
        return (rho == 0) and (alpha + beta == gamma)

    @classmethod
    def _marginal_product_capital(cls, capital, fossil_fuel, tfp, alpha, beta,
                                  gamma, sigma):
        """Non-renewable sector marginal product of capital."""
        if cls._is_cobb_douglas(alpha, beta, gamma, sigma):
            mpk = tfp * alpha * K**(alpha - 1) * F**beta
        else:
            mpk = tfp * alpha * gamma * K**(rho - 1) * (alpha * K**rho + beta * F**rho)**((gamma / rho) - 1)
        assert mpk > 0, "Marginal product of capital is not positive!"
        return mpk

    @staticmethod
    def _percentage_adjustment_costs(capital, investment, phi, **params):
        """Convex capital adjustment cost function."""
        costs = (phi / 2) * (investment / capital)**2
        assert costs > 0, "Non-renewable sector adjustment costs are not positive!"
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
