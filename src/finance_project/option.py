from dataclasses import dataclass

@dataclass()
class Option:
    strike: float
    maturity: float
    type: str = "call"  # call or put
    exercise: str = "european"  # european, american, bermudan
    exercise_steps: set[int] | None = None  # only for bermudan option