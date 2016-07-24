from scipy import optimize

from energy_consumers import Consumer
from energy_sectors import NonRenewableEnergySector, RenewableEnergySector

class EnergyMarket:

    @classmethod
    def find_market_price(cls, non_renewable_capital, consumer_params, non_renewable_params, renewable_params):
        """Use root finding algorith to determine the market price."""
        args = (non_renewable_capital, consumer_params, non_renewable_params, renewable_params)
        price, results = optimize.brentq(cls._excess_demand, 1e-12, 1e12, args, full_output=True)
        return price if results.converged else results

    @staticmethod
    def _aggregate_demand(price, params):
        return Consumer.demand(price, params)

    @classmethod
    def _aggregate_supply(cls, price, non_renewable_capital, non_renewable_params, renewable_params):
        """Aggregate energy supply is total energy produced by the non-renewable and renewable sectors."""
        aggregate_supply = (cls._non_renewable_sector_supply(price, non_renewable_capital, non_renewable_params) +
                            cls._renewable_sector_supply(price, renewable_params))
        return aggregate_supply

    @staticmethod
    def _renewable_sector_supply(price, params):
        capital = RenewableEnergySector.capital_demand(price, params)
        return RenewableEnergySector.output(capital, params)

    @staticmethod
    def _non_renewable_sector_supply(price, capital, params):
        fossil_fuel = NonRenewableEnergySector.fossil_fuel_demand(capital, price, params)
        return NonRenewableEnergySector.output(capital, fossil_fuel, params)

    @classmethod
    def _excess_demand(cls, price, non_renewable_capital, consumer_params, non_renewable_params, renewable_params):
        """
        Excess demand for energy is the difference between the total quantity of energy required by consumers and the total
        amount of energy supplied by the non-renewable and renewable energy sectors. The root of this function is the market price for energy.

        """
        excess_demand = (cls._aggregate_demand(price, consumer_params) -
                         cls._aggregate_supply(price, non_renewable_capital, non_renewable_params, renewable_params))
        return excess_demand
