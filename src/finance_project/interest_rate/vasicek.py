from finance_project.interest_rate.IRModel import InterestRateModel
import numpy as np

class VasicekModel(InterestRateModel):

    def __init__(self, initial_rate, volatility, mean_level, mean_revert_speed):
        super().__init__(initial_rate)
        self.volatility = volatility
        self.mean_level = mean_level
        self.mean_revert_speed = mean_revert_speed

    def drift(self, t, r):
        return self.mean_revert_speed * (self.mean_level - r)

    def diffusion(self, t, r):
        return self.volatility

    def simulate_paths(self, maturity: float, num_steps: int, num_paths: int, seed: int = 42, method="euler"):
        if method == "euler":
            return super().simulate_paths(maturity, num_steps, num_paths, seed)

        if method == "exact":
            dt = maturity / num_steps
            paths = np.zeros((num_paths, num_steps + 1))
            paths[:, 0] = self.initial_rate
            rng = np.random.default_rng(seed)

            for step in range(num_steps):
                cur_rates = paths[:, step]
                z = rng.standard_normal(num_paths)

                paths[:, step + 1] = self.mean_level + (cur_rates - self.mean_level) * np.exp(
                    -self.mean_revert_speed * dt) + self.volatility * np.sqrt(
                    (1 - np.exp(-2 * self.mean_revert_speed * dt)) / (2 * self.mean_revert_speed)) * z

            return paths

        raise ValueError("Unknown simulation method")
