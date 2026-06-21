from dataclasses import dataclass
import numpy as np
from .option import Option


@dataclass
class TreeParameters:
    time_step: float
    up: float
    down: float
    risk_neutral_prob: float


class BinomialTreeModel:
    def __init__(
            self,
            init_price: float,
            rate: float,
            volatility: float,
            num_of_steps: int,
            model: str = "crr"
    ):
        self.init_price = init_price
        self.rate = rate
        self.volatility = volatility
        self.num_of_steps = num_of_steps
        self.model = model

    def _compute_params(self, maturity: float) -> TreeParameters:
        dt = maturity / self.num_of_steps

        if self.model == "crr":
            up = np.exp(self.volatility * np.sqrt(dt))
            down = np.exp(-self.volatility * np.sqrt(dt))
            risk_neutral_prob = (np.exp(self.rate * dt) - down) / (up - down)

            if risk_neutral_prob > 1 or risk_neutral_prob < 0:
                raise ValueError("risk_neutral_prob is not between 0 and 1")

        elif self.model == "jarrow-rudd":
            up = np.exp((self.rate - self.volatility ** 2 / 2) * dt + self.volatility * np.sqrt(dt))
            down = np.exp((self.rate - self.volatility ** 2 / 2) * dt - self.volatility * np.sqrt(dt))
            risk_neutral_prob = 0.5

        else:
            raise ValueError("model must be 'crr' or 'jarrow-rudd'")

        return TreeParameters(dt, up, down, risk_neutral_prob)

    def _build_tree(self, params: TreeParameters) -> list[list[float]]:
        up = params.up
        down = params.down
        tree = [[self.init_price]]
        for step in range(self.num_of_steps):
            new_row = [prev * down for prev in tree[-1]]
            new_row.append(tree[-1][-1] * up)
            tree.append(new_row)

        return tree

    @staticmethod
    def _compute_payoff(prices: list[float], option: Option) -> list[float]:
        if option.type == "call":
            return [max(0.0, price - option.strike) for price in prices]
        elif option.type == "put":
            return [max(0.0, option.strike - price) for price in prices]
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def _backward_induction(self,
                           tree: list[list[float]],
                           params: TreeParameters,
                           option: Option
                           ) -> float:
        values = self._compute_payoff(tree[-1], option)

        for i in range(self.num_of_steps, 0, -1):
            dt = params.time_step
            prob = params.risk_neutral_prob

            values = [
                np.exp(-self.rate * dt)
                * (prob * values[j + 1] + (1 - prob) * values[j])
                for j in range(i)
            ]

            if (option.exercise == "american" or
                    (option.exercise == "bermudan"
                     and option.exercise_steps is not None
                     and (i - 1) in option.exercise_steps)
            ):
                values = [
                    max(value, cur_payoff)
                    for value, cur_payoff in zip(values, self._compute_payoff(tree[i - 1], option))
                ]

        return values[0]

    def price(self, option: Option) -> float:
        params = self._compute_params(option.maturity)
        tree = self._build_tree(params)

        price = self._backward_induction(tree, params, option)

        return price
