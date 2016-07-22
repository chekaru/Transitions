class RenewableEnergySector:

    @staticmethod
    def output(capital, alpha, tfp, **params):
        """Renewable energy output."""
        return tfp * capital**alpha

    @classmethod
    def revenue(cls, capital, alpha, energy_price, tfp, **params):
        """Renewable energy revenue."""
        return cls.subsidy(energy_price) * cls.output(capital, alpha, tfp, **params)

    @staticmethod
    def subsidy(energy_price, **params):
        """For now assume that subsidy is the energy price."""
        return energy_price

    @staticmethod
    def total_costs(capital, gross_interest_rate, **params):
        """Renewable energy production costs."""
        return gross_interest_rate * capital

    @classmethod
    def profits(cls, capital, energy_price, alpha, gross_interest_rate, tfp, **params):
        """Renewable energy sector profits."""
        pi = (cls.revenue(capital, alpha, energy_price, tfp, **params) -
              cls.total_costs(capital, gross_interest_rate, **params))
        return pi

    @classmethod
    def capital_demand(cls, alpha, energy_price, gross_interest_rate, tfp, **params):
        """
        Renewable energy sector demand for physical capital.

        Note
        ----
        The demand for physical capital is a function of the price of energy
        (which may or may not be subsidized) and model parameters.

        """
        demand = (alpha * tfp * (cls.subsidy(energy_price) / gross_interest_rate))**(1 - alpha)
        assert demand > 0, "Renewable energy sector capital demand of {} is not positive!".format(demand)
        return demand


class NonRenewableEnergySector:

    @staticmethod
    def output(capital, fossil_fuel, beta, tfp, **params):
        """Non-renewable energy output."""
        energy = tfp * capital**beta * fossil_fuel**(1 - beta)
        assert energy > 0, "Non-renewable energy output of {} is not positive!".format(energy)
        return energy

    @classmethod
    def profits(cls, capital, previous_capital, fossil_fuel, beta, energy_price, fossil_fuel_price, phi, tfp):
        """Profits are difference between revenue and costs."""
        pi = (cls.revenue(capital, fossil_fuel, beta, energy_price, tfp) -
              cls.total_costs(capital, previous_capital, fossil_fuel, fossil_fuel_price, phi))
        return pi

    @classmethod
    def revenue(cls, capital, fossil_fuel, beta, energy_price, tfp, **params):
        """Renewable energy revenue."""
        return energy_price * cls.output(capital, fossil_fuel, beta, tfp, **params)

    @classmethod
    def total_costs(cls, capital, previous_capital, fossil_fuel, fossil_fuel_price, phi):
        """Total costs of producing energy from fossil fuels."""
        costs = (cls.total_cost_capital(capital, previous_capital, phi) +
                 cls.total_cost_fossil_fuel(fossil_fuel, fossil_fuel_price))
        assert costs > 0, "Non-renewable energy sector total costs of {} are not positive!".format(costs)
        return costs

    @classmethod
    def total_cost_capital(cls, capital, previous_capital, phi):
        """Total costs of capital for use in producing energy from fossil fuels."""
        return gross_interest_rate * capital + cls.capital_adjustment_costs(capital, previous_capital, phi)

    @classmethod
    def total_cost_fossil_fuel(cls, fossil_fuel, fossil_fuel_price):
        """Total costs of purchasing fossil fuel for use in energy production."""
        return fossil_fuel_price * fossil_fuel

    @staticmethod
    def capital_adjustment_costs(capital, previous_capital, phi):
        """Convex capital adjustment cost function."""
        return (phi / 2) * (capital - previous_capital)**2

    @classmethod
    def marginal_cost_capital(cls, capital, previous_capital, gross_interest_rate, phi):
        """Costs of adjusting stock of physical capital."""
        marginal_cost = gross_interest_rate + cls.marginal_capital_adjustment_costs(capital, previous_capital, phi)
        assert marginal_cost > 0, "Non-renewable energy sector marginal cost of capital of {} is not positive!".format(marginal_cost)
        return marginal_cost

    @staticmethod
    def marginal_capital_adjustment_costs(capital, previous_capital, phi):
        """Marginal costs of adjusting stock of capital."""
        return phi * (capital - previous_capital)

    @classmethod
    def capital_output_ratio(cls, capital, fossil_fuel, beta, tfp, **params):
        """Ratio of capital stock to output."""
        ratio =  capital / cls.output(capital, fossil_fuel, beta, tfp)
        assert ratio > 0, "Non-renewable energy sector capital-output ratio of {} is not positive!".format(ratio)
        return ratio

    @classmethod
    def marginal_product_capital(cls, capital, fossil_fuel, beta, energy_price, tfp, **params):
        """Marginal product of installed capital."""
        marginal_product = beta / cls.capital_output_ratio(capital, fossil_fuel, beta, tfp, **params)
        assert marginal_product > 0, "Non-renewable energy sector marginal product of capital of {} is not positive!".format(marginal_product)
        return marginal_product

    @classmethod
    def value_marginal_product_capital(cls, capital, fossil_fuel, beta, energy_price, tfp, **params):
        """Contribution to firm revenue of the marginal unit of installed capital."""
        vmp = energy_price * cls.marginal_product_capital(capital, fossil_fuel, beta, energy_price, tfp, **params)
        assert vmp > 0, "Non-renewable energy sector value marginal product of capital of {} is not positive!".format(vmp)
        return vmp

    @classmethod
    def net_value_marginal_product_capital(cls, capital, previous_capital, fossil_fuel, beta, energy_price, gross_interest_rate, phi, tfp, **params):
        """Value marginal product of capital less marginal costs of capital."""
        return cls.value_marginal_product_capital(capital, fossil_fuel, beta, energy_price, tfp, **params) - cls.marginal_cost_capital(capital, previous_capital, gross_interest_rate, phi)

    @classmethod
    def future_capital_demand(cls, capital, previous_capital, fossil_fuel, beta, energy_price, eta, gross_interest_rate, phi, tfp, **params):
        """Demand for capital is forward looking?"""
        demand = capital + (cls.net_value_marginal_product_capital(capital, previous_capital, fossil_fuel, beta, energy_price, gross_interest_rate, phi, tfp, **params) / (eta * phi))
        assert demand > 0, "Non-renewable energy sector capital demand of {} is not positive!".format(demand)
        return demand

    @staticmethod
    def fossil_fuel_demand(capital, beta, energy_price, fossil_fuel_price, tfp, **params):
        """Non-renewable energy sector demand for fossil fuels."""
        demand = (tfp * (1 - beta) * (energy_price / fossil_fuel_price))**(1 / beta) * capital
        assert demand > 0, "Non-renewable energy sector fossil fuel demand of {} is not positive!".format(demand)
        return demand
