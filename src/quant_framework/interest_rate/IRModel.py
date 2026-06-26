from abc import ABC, abstractmethod
import numpy as np

class InterestRateModel(ABC):
    def __init__(self, initial_rate: float):
        self.initial_rate = initial_rate

    @abstractmethod
    def drift(self, t, r):
        pass

    @abstractmethod
    def diffusion(self, t, r):
        pass

    def simulate_paths(self, maturity: float, num_steps: int, num_paths: int, seed: int = 42):
        dt = maturity / num_steps
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.initial_rate
        rng = np.random.default_rng(seed)

        for step in range(num_steps):
            cur_rates = paths[:, step]
            z = rng.standard_normal(num_paths)

            paths[:, step + 1] = cur_rates + self.drift(dt * step, cur_rates) * dt + self.diffusion(
                dt * step, cur_rates) * np.sqrt(dt) * z

        return paths
