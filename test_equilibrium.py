"""
Confirms that the computed long-run equilibrium matches the derived, analytic
solution for the equilibrium when there is fixed wholesale demand, Cobb-Douglas
non-renewable energy production and the subsidy paid to renewable energy
production is a constant markup over the wholesale market price.

"""
from energy_consumers import EnergyConsumer
from energy_markets import WholesaleEnergyMarket
from energy_sectors import NonRenewableEnergySector, RenewableEnergySector
from models import TransitionDynamicsModel
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


def test_equilibrium_capital():
    """Compare the computed equilbrium capital value with is analytic solution."""
    prices = (CAPITAL_PRICE, FOSSIL_FUEL_PRICE, INTEREST_RATE)
    model = TransitionDynamicsModel(ENERGY_MARKET, *prices)
    _, numeric_capital = model.equilibrium
    analytic_capital = _equilibrium_capital(ENERGY_MARKET, *prices)
    rel_error = abs(analytic_capital - numeric_capital) / analytic_capital
    assert rel_error <= 1e-12


def _equilibrium_capital(energy_market, capital_price, fossil_fuel_price, interest_rate):
    alpha = energy_market.renewable_sector._alpha
    delta_R = energy_market.renewable_sector._delta
    mu_R = energy_market.renewable_sector._mu
    tfp_R = energy_market.renewable_sector._tfp
    tfp_NR = energy_market.non_renewable_sector._tfp
    quantity_demand = energy_market.consumer._quantity_demand
    energy_price = _equilibrium_energy_price(energy_market, capital_price, fossil_fuel_price, interest_rate)
    relative_price_1 = energy_price / fossil_fuel_price
    relative_price_2 = fossil_fuel_price / capital_price
    capital = ((quantity_demand / (tfp_NR**(1 / (1 - alpha)) * (alpha * relative_price_1)**(alpha / (1 - alpha)))) -
               (tfp_R / tfp_NR)**(1 / (1 - alpha)) * (relative_price_2 * ((1 + mu_R) / (interest_rate + delta_R)))**(alpha / (1 - alpha)) )
    return capital


def _equilibrium_energy_price(energy_market, capital_price, fossil_fuel_price, interest_rate):
    """Equilibrium wholesale market price of energy."""
    alpha = energy_market.renewable_sector._alpha
    delta_NR = energy_market.non_renewable_sector._delta
    phi = energy_market.non_renewable_sector._phi
    tfp_NR = energy_market.non_renewable_sector._tfp
    q = energy_market.non_renewable_sector.equilibrium_q
    relative_price = capital_price / fossil_fuel_price
    price = alpha**-alpha * (fossil_fuel_price / tfp_NR) * (relative_price * (((interest_rate + delta_NR) * q - phi * delta_NR**3) / (1 - alpha)))**(1 - alpha)
    return price
