import numpy as np
from scipy.stats import norm
from .option import Option


class BlackScholesModel():
    def __init__(self,
            init_price: float,
            rate: float,
            volatility: float
    ):
        self.init_price = init_price
        self.rate = rate
        self.volatility = volatility

    def _d1(self, option: Option) -> float:
        d1 = ((np.log(self.init_price / option.strike) + (self.rate + self.volatility**2 / 2) * option.maturity) /
              (self.volatility * np.sqrt(option.maturity)))

        return d1

    def _d2(self, option: Option) -> float:
        d2 = ((np.log(self.init_price / option.strike) + (self.rate - self.volatility ** 2 / 2) * option.maturity) /
              (self.volatility * np.sqrt(option.maturity)))

        return d2

    def price(self, option: Option) -> float:
        if option.exercise != "european":
            raise ValueError("Black-Scholes formula supports only European options.")

        d1 = self._d1(option)
        d2 = self._d2(option)

        if option.type == 'call':
            price = self.init_price * norm.cdf(d1) - option.strike * np.exp(-self.rate * option.maturity) * norm.cdf(d2)

        elif option.type == 'put':
            price = -self.init_price * norm.cdf(-d1) + option.strike * np.exp(-self.rate * option.maturity) * norm.cdf(-d2)

        else:
            raise ValueError("Option type must be 'call' or 'put'")

        return price