from finance_project.option import Option
from finance_project.black_scholes import BlackScholesModel
from finance_project.binomial_tree import BinomialTreeModel

def test_black_scholes_call_value():
    option = Option(strike=100, maturity=1, type="call")
    model = BlackScholesModel(init_price=100, rate=0.05, volatility=0.2)

    price = model.price(option)

    assert abs(price - 10.4506) < 1e-4


def test_black_scholes_put_value():
    option = Option(strike=100, maturity=1, type="put")
    model = BlackScholesModel(init_price=100, rate=0.05, volatility=0.2)

    price = model.price(option)

    assert abs(price - 5.5735) < 1e-4


def test_crr_converges_to_black_scholes_call():
    option = Option(strike=100, maturity=1, type="call")

    bs = BlackScholesModel(init_price=100, rate=0.05, volatility=0.2)
    bs_price = bs.price(option)

    crr = BinomialTreeModel(
        init_price=100,
        rate=0.05,
        volatility=0.2,
        num_of_steps=1000,
        model="crr",
    )

    crr_price = crr.price(option)

    assert abs(crr_price - bs_price) < 0.01