from typing import Any, MutableMapping

import numpy as np

from ..problem import BaseProblem
from ..utils import Binary2Float, l2_residual
from .utils import discretize_matrix, get_bit_value, get_qubo


class BlLstsqProblem(BaseProblem):

    _solver_type = "direct"
    _default_qubo_params = {
        "direct": {
            "num_bits": 4,
            "fixed_point": 1,
            "sign": "pn",
            "eq_scaling_val": 1 / 8,
        },
        "iterative": {
            "num_bits": 2,
            "fixed_point": 1,
            "sign": "p",
            "eq_scaling_val": 1 / 8,
        },
    }

    def __init__(self, problem_params: MutableMapping[str, Any], qubo_params: MutableMapping[str, Any] = {}) -> None:
        self._user_qubo_params = qubo_params
        super().__init__(problem_params=problem_params, qubo_params=qubo_params)
        self._check_problem_params(self.problem_params)
        self._check_qubo_params(self.qubo_params)

    @property
    def default_problem_params(self):
        return {"obj_func": l2_residual}

    @property
    def default_qubo_params(self):
        return self._default_qubo_params[self._solver_type]

    @property
    def required_problem_params(self):
        return ["A", "b"]

    @property
    def required_qubo_params(self):
        return ["num_bits", "fixed_point", "sign"]

    @property
    def A(self):
        return self.problem_params["A"]

    @A.setter
    def A(self, val):
        if val.shape[0] != self.problem_params["A"].size:
            raise ValueError("The number of rows of matrix `A` must be the same as" " the length of the vector `b`.")
        self.problem_params["A"] = val

    @property
    def b(self):
        return self.problem_params["b"]

    @b.setter
    def b(self, val):
        if val.size != self.problem_params["A"].shape[0]:
            raise ValueError("The length of the vector `b` must be the same as" " the number of rows of matrix `A`.")
        self.problem_params["b"] = val

    @property
    def A_discrete(self):
        return discretize_matrix(self.A, self.bit_value)

    @property
    def bit_value(self):
        return get_bit_value(
            self.qubo_params["num_bits"], fixed_point=self.qubo_params["fixed_point"], sign=self.qubo_params["sign"]
        )

    def get_qubo(self, return_matrix: bool = True):
        return get_qubo(
            self.A_discrete, self.b, eq_scaling_val=self.qubo_params["eq_scaling_val"], return_matrix=return_matrix
        )

    def calc_qubo_obj(self, binary_array: np.ndarray):
        return binary_array.T @ self.get_qubo() @ binary_array

    def calc_obj(self, array: np.ndarray, binary=False) -> float:
        return (
            self.problem_params["obj_func"](
                self.problem_params["A"] @ Binary2Float.to_fixed_point(array, self.bit_value),
                self.problem_params["b"],
            )
            if binary
            else self.problem_params["obj_func"](
                self.problem_params["A"] @ array,
                self.problem_params["b"],
            )
        )
