from collections import defaultdict

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dwaveutils import bl_lstsq


@pytest.fixture
def A():
    return np.identity(3)


@pytest.fixture
def b():
    return np.array([0.75, 0.5, -0.5])


@pytest.fixture
def bit_value():
    return np.array([-1 * 2 ** 0, 2 ** -1, 2 ** -2])


@pytest.fixture
def A_discrete():
    return np.array(
        [
            [-1, 0.5, 0.25, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0.5, 0.25, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0.5, 0.25],
        ]
    )


def test_discretize_matrix(A, A_discrete, bit_value):
    """Test `discretize_matrix` function."""
    A_discrete_for_test = bl_lstsq.discretize_matrix(A, bit_value)
    assert_array_equal(A_discrete_for_test, A_discrete, err_msg="Wrong discretized")


@pytest.mark.parametrize(
    "num_bits, fixed_point, sign, expected_result",
    [
        (5, 0, "pn", np.array([-1, 0.5, 0.25, 0.125, 0.0625])),
        (5, 2, "p", np.array([2, 1, 0.5, 0.25, 0.125])),
        (5, 1, "n", np.array([-1, -0.5, -0.25, -0.125, -0.0625])),
    ],
)
def test_get_bit_value(num_bits, fixed_point, sign, expected_result):
    assert_array_equal(
        bl_lstsq.get_bit_value(num_bits, fixed_point=fixed_point, sign=sign),
        expected_result,
        err_msg="Wrong bit value",
    )


def test_bruteforce(A_discrete, b, bit_value):
    """Test `bruteforce` function."""
    best_q, best_x, min_norm = bl_lstsq.bruteforce(A_discrete, b, bit_value)
    expected_q = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0])
    expected_x = np.array([0.75, 0.5, -0.5])
    expected_norm = 0
    assert_array_equal(best_q, expected_q, err_msg="Wrong q")
    assert_array_equal(best_x, expected_x, err_msg="Wrong x")
    assert min_norm == expected_norm


def test_get_qubo(A_discrete, b):
    """Test `get_qubo` function."""
    Q = bl_lstsq.get_qubo(A_discrete, b, eq_scaling_val=1 / 2)
    expected_Q = defaultdict(
        int,
        {
            (0, 0): 1.25,
            (1, 1): -0.25,
            (1, 0): -0.5,
            (2, 2): -0.15625,
            (2, 0): -0.25,
            (2, 1): 0.125,
            (3, 3): 1.0,
            (4, 4): -0.125,
            (4, 3): -0.5,
            (5, 5): -0.09375,
            (5, 3): -0.25,
            (5, 4): 0.125,
            (6, 6): 0.0,
            (7, 7): 0.375,
            (7, 6): -0.5,
            (8, 8): 0.15625,
            (8, 6): -0.25,
            (8, 7): 0.125,
        },
    )
    assert Q == expected_Q, "Wrong QUBO"


if __name__ == "__main__":
    pytest.main([__file__])
