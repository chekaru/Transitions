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

    @staticmethod
    def output(capital, fossil_fuel, alpha, beta, gamma, sigma=1, **params):
        """
        Non-renewable sector sector energy output.

        Notes
        -----
        Sector uses a generalized version of the constant elasticity of
        substitution (CES) functional form.

        """
        rho = (sigma - 1) / sigma
        if (rho == 0) and (alpha + beta == gamma):
            energy = tfp * K**alpha * F**beta
        else:
            energy = tfp * (alpha * K**rho + beta * F**rho)**(gamma / rho)
        assert energy > 0, "Non-renewable energy output of {} is not positive!".format(energy)
        return energy

    @classmethod
    def profits(cls, capital, previous_capital, fossil_fuel, energy_price, params):
        """Profits are difference between revenue and costs."""
        pi = (cls.revenue(capital, fossil_fuel, energy_price, params) -
              cls.total_costs(capital, previous_capital, fossil_fuel, params))
        return pi

    @classmethod
    def revenue(cls, capital, fossil_fuel, energy_price, params):
        """Renewable energy revenue."""
        return energy_price * cls.output(capital, fossil_fuel, params)

    @classmethod
    def total_costs(cls, capital, previous_capital, fossil_fuel, params):
        """Total costs of producing energy from fossil fuels."""
        costs = (cls.total_cost_capital(capital, previous_capital, params) +
                 cls.total_cost_fossil_fuel(fossil_fuel, params))
        assert costs > 0, "Non-renewable energy sector total costs of {} are not positive!".format(costs)
        return costs

    @classmethod
    def total_cost_capital(cls, capital, previous_capital, params):
        """Total costs of capital for use in producing energy from fossil fuels."""
        costs = (params['gross_interest_rate'] * capital +
                 cls.capital_adjustment_costs(capital, previous_capital, params))
        assert costs > 0, "Non-renewable energy sector total costs of capital ]{} are not positive!".format(costs)
        return costs

    @classmethod
    def total_cost_fossil_fuel(cls, fossil_fuel, params):
        """Total costs of purchasing fossil fuel for use in energy production."""
        return params['fossil_fuel_price'] * fossil_fuel

    @staticmethod
    def capital_adjustment_costs(capital, previous_capital, params):
        """Convex capital adjustment cost function."""
        return (params['phi'] / 2) * (capital - previous_capital)**2

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
    def marginal_product_capital(cls, capital, fossil_fuel, energy_price, params):
        """Marginal product of installed capital."""
        mp = params['beta'] / cls.capital_output_ratio(capital, fossil_fuel, params)
        assert mp > 0, "Non-renewable energy sector marginal product of capital of {} is not positive!".format(mp)
        return mp

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
