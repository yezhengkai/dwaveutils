from typing import Any, MutableMapping

import numpy as np
import scipy.sparse as sp

from ..problem import BaseProblem
from ..utils import residual_sum_squares
from .utils import QUBO, fwd_modeling


class BinaryInverseProblem(BaseProblem):
    def __init__(self, problem_params: MutableMapping[str, Any]) -> None:
        super().__init__(problem_params=problem_params)
        self._check_problem_params(problem_params)
        self.qubo = QUBO(
            self.problem_params["fwd_model_func"],
            self.problem_params["obs_resp"],
            self.problem_params["low_high"],
            params_inv2fwd_func=self.problem_params["params_inv2fwd_func"],
            resp_all2meas_func=self.problem_params["resp_all2meas_func"],
        )

    @property
    def required_problem_params(self):
        return ["fwd_model_func", "obs_resp", "low_high"]

    @property
    def default_problem_params(self):
        return {"params_inv2fwd_func": None, "resp_all2meas_func": None, "obj_func": residual_sum_squares}

    @property
    def fwd_model_func(self):
        return self.problem_params["fwd_model_func"]

    @property
    def obs_resp(self):
        return self.problem_params["obs_resp"]

    @property
    def low_high(self):
        return self.problem_params["low_high"]

    @property
    def obj_func(self):
        return self.problem_params["obj_func"]

    def get_qubo(self, bin_model_params: np.ndarray, return_matrix: bool = True) -> sp.dok_matrix:
        return self.qubo.get(bin_model_params, return_matrix=return_matrix)

    def calc_qubo_obj(self, bin_model_params: np.ndarray):
        return bin_model_params.T @ self.get_qubo(bin_model_params) @ bin_model_params

    def calc_obj(self, bin_model_params: np.ndarray) -> float:
        return self.problem_params["obj_func"](
            self.fwd_modeling(bin_model_params),
            self.problem_params["obs_resp"],
        )

    def fwd_modeling(self, bin_model_params: np.ndarray):
        return fwd_modeling(
            self.problem_params["fwd_model_func"],
            bin_model_params,
            low_high=self.problem_params["low_high"],
            params_inv2fwd_func=self.problem_params["params_inv2fwd_func"],
            resp_all2meas_func=self.problem_params["resp_all2meas_func"],
        )
