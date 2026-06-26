import numpy as np
from finance_project.interest_rate.CIR import CIRModel
from finance_project.interest_rate.vasicek import VasicekModel


def test_cir_paths_shape():
    model = CIRModel(
        initial_rate=0.03,
        volatility=0.01,
        mean_level=0.05,
        mean_revert_speed=0.7,
    )

    paths = model.simulate_paths(
        maturity=1.0,
        num_steps=100,
        num_paths=50,
    )

    assert paths.shape == (50, 101)


def test_vasicek_initial_rate_is_first_column():
    model = VasicekModel(
        initial_rate=0.03,
        volatility=0.05,
        mean_level=0.05,
        mean_revert_speed=0.5,
    )

    paths = model.simulate_paths(1.0, 100, 50)

    assert np.allclose(paths[:, 0], 0.03)



def test_vasicek_seed_reproducibility():
    model = VasicekModel(
        initial_rate=0.03,
        volatility=0.01,
        mean_level=0.05,
        mean_revert_speed=0.5,
    )

    paths_1 = model.simulate_paths(1.0, 100, 50, seed=42)
    paths_2 = model.simulate_paths(1.0, 100, 50, seed=42)

    assert np.allclose(paths_1, paths_2)


def test_cir_seed_reproducibility():
    model = CIRModel(
        initial_rate=0.03,
        volatility=0.01,
        mean_level=0.05,
        mean_revert_speed=0.5,
    )

    paths_1 = model.simulate_paths(1.0, 100, 50, seed=42)
    paths_2 = model.simulate_paths(1.0, 100, 50, seed=42)

    assert np.allclose(paths_1, paths_2)


def test_cir_paths_are_non_negative():
    model = CIRModel(
        initial_rate=0.03,
        volatility=0.2,
        mean_level=0.05,
        mean_revert_speed=0.7,
    )

    paths = model.simulate_paths(
        maturity=1.0,
        num_steps=100,
        num_paths=500,
        seed=42,
    )

    assert np.all(paths >= 0.0)
