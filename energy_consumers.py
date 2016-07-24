class Consumer:

    @staticmethod
    def demand(energy_price, params):
        """For now just assume inelastic demand for energy."""
        return params['energy_quantity']
