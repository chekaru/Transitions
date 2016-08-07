class EnergyConsumer:

    def __init__(self, quantity_demand):
        self._quantity_demand = quantity_demand

    def demand(self, energy_price):
        """For now just assume inelastic demand for energy."""
        return self._quantity_demand
