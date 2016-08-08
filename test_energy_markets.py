"""
Confirms that the computed wholesale market price for energy matches the derived
analytic solution when there is fixed wholesale demand, Cobb-Douglas non-renewable
energy production and the subsidy paid to renewable energy production is a
constant markup over the wholesale market price.

"""
from energy_consumers import EnergyConsumer
from energy_markets import WholesaleEnergyMarket
from energy_sectors import NonRenewableEnergySector, RenewableEnergySector
import utils

# define energy sectors, consumers, and market as globals
RENEWABLE_SEED, RENEWABLE_SECTOR_PARAMS = utils.generate_renewable_sector_params()
RENEWABLE_SECTOR = RenewableEnergySector(**RENEWABLE_SECTOR_PARAMS)

NON_RENEWABLE_SEED, NON_RENEWABLE_SECTOR_PARAMS = utils.generate_non_renewable_sector_params(RENEWABLE_SECTOR_PARAMS)
NON_RENEWABLE_SECTOR= NonRenewableEnergySector(**NON_RENEWABLE_SECTOR_PARAMS)

CONSUMER_SEED, CONSUMER_PARAMS = utils.generate_consumer_params()
CONSUMER = EnergyConsumer(**CONSUMER_PARAMS)

ENERGY_MARKET = WholesaleEnergyMarket(CONSUMER, NON_RENEWABLE_SECTOR, RENEWABLE_SECTOR)

# specify exogenous prices as global variables
PRICES_SEED, PRICES = utils.generate_prices()
CAPITAL_PRICE, FOSSIL_FUEL_PRICE, INTEREST_RATE = PRICES


def test_wholesale_market_price():
    """Compare the computed wholesale market price with the analytic solution."""
    prices = (CAPITAL_PRICE, FOSSIL_FUEL_PRICE, INTEREST_RATE)
    capital = 10  # this should be cleverly chosen!
    analytic_result = _energy_market_price(capital, ENERGY_MARKET, *prices)
    numeric_result = ENERGY_MARKET.find_market_price(capital, *prices)
    abs_error = abs(analytic_result - numeric_result)
    assert abs_error <= 1e-12


def _energy_market_price(capital, energy_market, capital_price, fossil_fuel_price, interest_rate):
    """Analytic solution for wholesale market price when alpha = alpha_R = 1 - alpha_NR."""
    quantity_demand = energy_market.consumer._quantity_demand
    tfp_NR = energy_market.non_renewable_sector._tfp
    tfp_R = energy_market.renewable_sector._tfp
    alpha = energy_market.renewable_sector._alpha
    delta_R = energy_market.renewable_sector._delta
    mu_R = energy_market.renewable_sector._mu
    denominator = (tfp_NR**(1 / (1 - alpha)) * (1 / fossil_fuel_price)**(alpha / (1 - alpha)) * capital +
                   tfp_R**(1 / (1 - alpha)) * (((1 + mu_R) / capital_price) * (1 / (interest_rate + delta_R)))**(alpha / (1 - alpha)))
    price = (1 / alpha) * (quantity_demand / denominator)**((1 - alpha) / alpha)
    return price
