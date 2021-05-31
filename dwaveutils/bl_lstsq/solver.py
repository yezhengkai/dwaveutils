import warnings
from collections import ChainMap
from copy import deepcopy
from typing import Any, Dict, MutableMapping, Optional

import dimod
import numpy as np

from ..solver import BaseDirectSolver, BaseIterativeSolver
from ..utils import Binary2Float


class BlLstsqDirectSolver(BaseDirectSolver):
    def __init__(
        self, problem, sampler: Optional[dimod.Sampler] = None, sampling_params: MutableMapping[str, Any] = {}
    ) -> None:
        super().__init__(problem, sampler=sampler, sampling_params=sampling_params)

        # redefine problem.default_qubo_params
        self.problem._solver_type = "direct"
        self.problem.qubo_params = dict(ChainMap(self.problem._user_qubo_params, self.problem.default_qubo_params))

    def solve(self, sampler: Optional[dimod.Sampler] = None, sampling_params: MutableMapping[str, Any] = {}):

        # update sampler, sampling_params, iter_params
        sampler = sampler if sampler is not None else self.sampler
        self._check_sampler(sampler)
        self.sampler = sampler
        self._check_sampling_params(sampling_params)
        self.sampling_params = dict(ChainMap(sampling_params, self.sampling_params, self.default_sampling_params))

        # sampling
        sampleset = self.sampler.sample_qubo(self.problem.get_qubo(), **self.sampling_params)  # type: ignore

        # recover x from q
        lowest_q = np.fromiter(sampleset.first.sample.values(), dtype=np.float64)
        x = Binary2Float.to_fixed_point(lowest_q, self.problem.bit_value)
        qubo_obj = self.problem.calc_qubo_obj(lowest_q)
        obj = self.problem.calc_obj(x)

        return {"x": x, "qubo_obj": qubo_obj, "obj": obj, "sampleset": sampleset}


class BlLstsqIterativeSolver(BaseIterativeSolver):
    def __init__(
        self,
        problem,
        sampler: Optional[dimod.Sampler] = None,
        sampling_params: MutableMapping[str, Any] = {},
        iter_params: MutableMapping[str, Any] = {},
    ) -> None:

        super().__init__(problem, sampler=sampler, sampling_params=sampling_params)
        self.iter_params = dict(ChainMap(iter_params, self.default_sampling_params))

        # redefine problem.default_qubo_params
        self.problem._solver_type = "iterative"
        self.problem.qubo_params = dict(ChainMap(self.problem._user_qubo_params, self.problem.default_qubo_params))

    @property
    def default_iter_params(self):
        return {"scale_factor": 2, "num_iter": 10, "obj_tol": 1e-3, "verbose": False}

    def _check_iter_params(self, iter_params):
        if not isinstance(iter_params, MutableMapping):
            raise TypeError(
                f"`iter_params` must be an instance of `MutableMapping`, not an instance of {type(iter_params)}"
            )
        if "num_iter" in iter_params:
            if iter_params["num_iter"] <= 0:
                iter_params["num_iter"] = self.default_iter_params["num_iter"]
                warnings.warn(f"Set num_iter = {iter_params['num_iter']}")

    def _show_verbose_iter(self, iter_num, x, qubo_obj, obj):
        with np.printoptions(precision=4, suppress=True):
            print(f"Iteration: {iter_num}")
            print(f"  - x: {x}")
            print(f"  - qubo_obj: {qubo_obj:.8e}")
            print(f"  - obj: {obj:.8e}")

    def solve(
        self,
        initial_x: np.ndarray,
        sampler: Optional[dimod.Sampler] = None,
        sampling_params: MutableMapping[str, Any] = {},
        iter_params: MutableMapping[str, Any] = {},
    ):

        # update sampler, sampling_params, iter_params
        sampler = sampler if sampler is not None else self.sampler
        self._check_sampler(sampler)
        self.sampler = sampler
        self._check_sampling_params(sampling_params)
        self.sampling_params = dict(ChainMap(sampling_params, self.sampling_params, self.default_sampling_params))
        self._check_iter_params(iter_params)
        self.iter_params = dict(ChainMap(iter_params, self.iter_params, self.default_iter_params))
        self.tmp_problem = deepcopy(self.problem)

        # parameters for iteration
        ones_vector = np.ones(self.problem.b.size)
        scale_factor = self.iter_params["scale_factor"]
        num_iter = self.iter_params["num_iter"]
        obj_tol = self.iter_params["obj_tol"]
        bit_value = self.problem.bit_value
        history: Dict[str, list] = {
            "x": [initial_x],
            "qubo_obj": [np.nan],
            "obj": [self.problem.calc_obj(initial_x)],
        }
        if num_iter <= 0:
            num_iter = 10
            warnings.warn("Set num_iter=10")

        if self.iter_params["verbose"]:
            self._show_verbose_iter(0, history["x"][0], history["qubo_obj"][0], history["obj"][0])

        # Start iteration
        for iter_num in range(num_iter):
            # construct new RHS vector
            tmp_b = (
                self.problem.b + scale_factor * (self.problem.A @ ones_vector) - self.problem.A @ initial_x
            ) / scale_factor

            # get qubo
            self.tmp_problem.b = tmp_b
            Q = self.tmp_problem.get_qubo()

            # sampling
            sampleset = self.sampler.sample_qubo(Q, **self.sampling_params)  # type: ignore

            # recover improvement vector from q
            lowest_q = np.fromiter(sampleset.first.sample.values(), dtype=np.float64)
            improvement_vector = Binary2Float.to_fixed_point(lowest_q, bit_value)

            # update initial guess
            improvement_x = scale_factor * (improvement_vector - ones_vector)
            x = initial_x + improvement_x
            qubo_obj = self.tmp_problem.calc_qubo_obj(lowest_q)
            obj = self.problem.calc_obj(x)
            history["x"].append(x)
            history["qubo_obj"].append(qubo_obj)
            history["obj"].append(obj)

            if self.iter_params["verbose"]:
                self._show_verbose_iter(iter_num + 1, x, qubo_obj, obj)

            if obj <= obj_tol:
                break
            else:
                initial_x = x
                # adjust scale factor
                if np.sum(np.abs(improvement_vector - 1) >= 0.5) == self.tmp_problem.b.size:
                    scale_factor /= 0.5
                elif np.sum(np.abs(improvement_vector - 1) >= 0.5) > self.tmp_problem.b.size // 2:
                    scale_factor /= 1
                elif np.sum(np.abs(improvement_vector - 1) <= 0.25) > self.tmp_problem.b.size // 2:
                    scale_factor /= 1.5
                else:
                    scale_factor /= 2

        # get the index of the smallest objective value from the history
        tmp_idx = int(np.argmin(history["obj"]))

        return {
            "x": history["x"][tmp_idx],
            "qubo_obj": history["qubo_obj"][tmp_idx],
            "obj": history["obj"][tmp_idx],
            "history": history,
        }
