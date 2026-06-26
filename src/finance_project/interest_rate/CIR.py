from finance_project.interest_rate.IRModel import InterestRateModel
import numpy as np

class CIRModel(InterestRateModel):

    def __init__(self, initial_rate, volatility, mean_level, mean_revert_speed):
        super().__init__(initial_rate)
        self.volatility = volatility
        self.mean_level = mean_level
        self.mean_revert_speed = mean_revert_speed

    def drift(self, t, r):
        return self.mean_revert_speed * (self.mean_level - r)

    def diffusion(self, t, r):
        return self.volatility * np.sqrt(np.maximum(r, 0.0))
