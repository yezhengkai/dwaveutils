import warnings
from collections import defaultdict
from typing import Any, Literal, Tuple, Union

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from ..utils import Binary2Float


def discretize_matrix(matrix: np.ndarray, bit_value: np.ndarray) -> np.ndarray:
    return np.kron(matrix, bit_value)


def get_bit_value(num_bits: int, fixed_point: int = 0, sign: Literal["pn", "p", "n"] = "pn") -> np.ndarray:
    """The value of each bit in two's-complement binary fixed-point numbers."""
    # 'pn': positive and negative value
    # 'p': only positive value
    # 'n': only negative value
    accepted_sign = ["pn", "p", "n"]
    if sign not in accepted_sign:
        warnings.warn("Use default `sign` setting.")
        sign = "pn"

    if sign == "pn":
        return np.array([-(2 ** fixed_point) if i == 0 else 2.0 ** (fixed_point - i) for i in range(0, num_bits)])
    elif sign == "p":
        return np.array([2.0 ** (fixed_point - i) for i in range(1, num_bits + 1)])
    else:
        return np.array([-(2.0 ** (fixed_point - i)) for i in range(1, num_bits + 1)])


def get_qubo(
    A_discrete: np.ndarray, b: np.ndarray, eq_scaling_val: float = 1 / 8, return_matrix: bool = False
) -> Union[defaultdict, sp.dok_matrix]:
    """Get coefficients of a quadratic unconstrained binary optimization (QUBO) problem defined by the dictionary."""

    # define weights
    # https://stackoverflow.com/questions/37524151/convert-a-deafultdict-to-numpy-matrix-or-a-csv-of-2d-matrix
    # https://scipy-lectures.org/advanced/scipy_sparse/dok_matrix.html
    qubo_a = np.diag(A_discrete.T @ A_discrete) - 2 * A_discrete.T @ b.flatten()
    qubo_b = sp.dok_matrix(np.tril(2 * A_discrete.T @ A_discrete, k=-1))

    # define objective
    if return_matrix:
        return eq_scaling_val * (sp.diags(qubo_a, format="dok") + qubo_b)
    else:
        Q = defaultdict(int, (eq_scaling_val * (sp.diags(qubo_a, format="dok") + qubo_b)).items())
        # force the diagonal entries to have values
        for i in range(qubo_a.size):
            Q[(i, i)] += 0.0
        return Q


def bruteforce(A_discrete: np.ndarray, b: np.ndarray, bit_value: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve A_discrete*q=b where q is a binary vector by brute force."""

    # number of predictor
    num_predictor_discrete = A_discrete.shape[1]
    # total number of solutions
    num_solution = 2 ** num_predictor_discrete
    # initialize best solution
    best_q: np.ndarray = np.nan * np.ones(num_predictor_discrete)
    # initialize minimum 2-norm
    min_norm = np.inf

    # loop all solutions
    with tqdm(range(num_solution), desc="brute force") as pbar:
        for i in pbar:
            # assign solution
            # https://stackoverflow.com/questions/13557937/how-to-convert-decimal-to-binary-list-in-python/13558001
            # https://stackoverflow.com/questions/13522773/convert-an-integer-to-binary-without-using-the-built-in-bin-function
            q = np.array([int(bit) for bit in format(i, f"0{num_predictor_discrete}b")])
            # calculate 2-norm
            new_norm: float = np.linalg.norm(A_discrete @ q - b, 2)
            # update best solution
            if new_norm < min_norm:
                min_norm = new_norm
                best_q = np.copy(q)

    if np.any(np.isnan(best_q)):
        raise ValueError("The `best_q` array has nan values.")
    else:
        best_x = Binary2Float.to_fixed_point(best_q, bit_value)
        return best_q, best_x, min_norm
