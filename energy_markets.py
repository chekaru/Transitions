from scipy import optimize


class WholesaleEnergyMarket:

    def __init__(self, consumer, non_renewable_sector, renewable_sector):
        self.consumer = consumer
        self.non_renewable_sector = non_renewable_sector
        self.renewable_sector = renewable_sector

    def find_market_price(self, capital, capital_price, fossil_fuel_price, interest_rate):
        """Use root finding algorithm to determine the market price."""
        args = (capital, capital_price, fossil_fuel_price, interest_rate)
        price, results = optimize.brentq(self._excess_demand, 1e-12, 1e12, args,
                                         full_output=True, xtol=1e-15)
        if results.converged:
            return price
        else:
            raise ValueError

    def _aggregate_demand(self, energy_price):
        return self.consumer.demand(energy_price)

    def _aggregate_supply(self, capital, capital_price, energy_price, fossil_fuel_price, interest_rate):
        """Aggregate energy supply is total energy produced by the non-renewable and renewable sectors."""
        supply = (self.non_renewable_sector.output(capital, energy_price, fossil_fuel_price) +
                  self.renewable_sector.output(capital_price, energy_price, interest_rate))
        return supply

    def _excess_demand(self, energy_price, capital, capital_price, fossil_fuel_price, interest_rate):
        """
        Excess demand for energy is the difference between the total quantity of energy required by consumers and the total
        amount of energy supplied by the non-renewable and renewable energy sectors. The root of this function is the market price for energy.

        """
        excess = (self._aggregate_demand(energy_price) -
                  self._aggregate_supply(capital, capital_price, energy_price, fossil_fuel_price, interest_rate))
        return excess
