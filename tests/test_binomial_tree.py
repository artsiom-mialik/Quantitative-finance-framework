from quant_framework.option import Option
from quant_framework.binomial_tree import BinomialTreeModel


def test_crr_bermudan_between_european_and_american():
    european = Option(
        strike=100,
        maturity=1,
        type="put",
        exercise="european",
    )

    bermudan = Option(
        strike=100,
        maturity=1,
        type="put",
        exercise="bermudan",
        exercise_steps={50, 100, 150, 200},
    )

    american = Option(
        strike=100,
        maturity=1,
        type="put",
        exercise="american",
    )

    model = BinomialTreeModel(
        init_price=100,
        rate=0.05,
        volatility=0.2,
        num_of_steps=200,
        model="crr",
    )

    european_price = model.price(european)
    bermudan_price = model.price(bermudan)
    american_price = model.price(american)

    assert european_price <= bermudan_price <= american_price


def test_jr_bermudan_between_european_and_american():
    european = Option(
        strike=100,
        maturity=1,
        type="put",
        exercise="european",
    )

    bermudan = Option(
        strike=100,
        maturity=1,
        type="put",
        exercise="bermudan",
        exercise_steps={50, 100, 150, 200},
    )

    american = Option(
        strike=100,
        maturity=1,
        type="put",
        exercise="american",
    )

    model = BinomialTreeModel(
        init_price=100,
        rate=0.05,
        volatility=0.2,
        num_of_steps=200,
        model="jarrow-rudd",
    )

    european_price = model.price(european)
    bermudan_price = model.price(bermudan)
    american_price = model.price(american)

    assert european_price <= bermudan_price <= american_price
