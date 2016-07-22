class Consumer:

    @staticmethod
    def demand(energy_price, energy_quantity, **params):
        """For now just assume inelastic demand for energy."""
        return energy_quantity
