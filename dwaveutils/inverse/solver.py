import warnings
from collections import ChainMap
from typing import Any, MutableMapping, Optional

import dimod
import numpy as np

from ..solver import BaseIterativeSolver
from .utils import flip_bits


class BinaryInverseIterativeSolver(BaseIterativeSolver):
    def __init__(
        self,
        problem,
        sampler: Optional[dimod.Sampler] = None,
        sampling_params: MutableMapping[str, Any] = {},
        iter_params: MutableMapping[str, Any] = {},
    ) -> None:
        super().__init__(problem, sampler=sampler, sampling_params=sampling_params)
        self.iter_params = dict(ChainMap(iter_params, self.default_sampling_params))

    @property
    def default_iter_params(self):
        return {
            "num_iter": 20,
            "obj_tol": 0,
            "check_num_obj": 48,
            "flip": True,
            "repeat_solution_stop_condition": False,
            "prob_random_obj": 0.0,
            "random_seed": None,
            "verbose": False,
        }

    def _check_iter_params(self, iter_params):
        if not isinstance(iter_params, MutableMapping):
            raise TypeError(
                f"`iter_params` must be an instance of `MutableMapping`, not an instance of {type(iter_params)}"
            )
        if "num_iter" in iter_params:
            if iter_params["num_iter"] <= 0:
                iter_params["num_iter"] = self.default_iter_params["num_iter"]
                warnings.warn(f"Set num_iter = {iter_params['num_iter']}")

    def _show_verbose_iter(self, iter_num, bin_model_params, qubo_obj, obj):
        print(f"Iteration: {iter_num}")
        print(f"  - bin_model_params = {bin_model_params}")
        print(f"  - qubo_obj = {qubo_obj:.8e}")
        print(f"  - obj = {obj:.8e}")

    def solve(
        self,
        initial_bin_model_params: np.ndarray,
        sampler: Optional[dimod.Sampler] = None,
        sampling_params: MutableMapping[str, Any] = {},
        iter_params: MutableMapping[str, Any] = {},
    ) -> dict:

        # update sampler, sampling_params, iter_params
        sampler = sampler if sampler is not None else self.sampler
        self._check_sampler(sampler)
        self.sampler = sampler
        self._check_sampling_params(sampling_params)
        self.sampling_params = dict(ChainMap(sampling_params, self.sampling_params, self.default_sampling_params))
        self._check_iter_params(iter_params)
        self.iter_params = dict(ChainMap(iter_params, self.iter_params, self.default_iter_params))

        # assign variables
        current_bin_model_params = np.copy(initial_bin_model_params)
        num_iter = self.iter_params["num_iter"]
        obj_tol = self.iter_params["obj_tol"]
        check_num_obj = self.iter_params["check_num_obj"]
        flip = self.iter_params["flip"]
        repeat_solution_stop_condition = self.iter_params["repeat_solution_stop_condition"]
        prob_random_obj = self.iter_params["prob_random_obj"]
        rng = (
            np.random.default_rng()
            if self.iter_params["random_seed"] is None
            else np.random.default_rng(self.iter_params["random_seed"])
        )
        verbose = self.iter_params["verbose"]
        history = {
            "bin_model_params": [initial_bin_model_params],
            "qubo_obj": [self.problem.calc_qubo_obj(initial_bin_model_params)],
            "obj": [self.problem.calc_obj(initial_bin_model_params)],
        }

        if verbose:
            self._show_verbose_iter(0, history["bin_model_params"][0], history["qubo_obj"][0], history["obj"][0])

        # start iteration
        for iter_num in range(num_iter):
            # get qubo
            Q = self.problem.get_qubo(current_bin_model_params, return_matrix=True)

            # get sampleset from sampler
            sampleset = self.sampler.sample_qubo(Q.toarray(), **self.sampling_params)  # type: ignore
            sampleset = sampleset.record.sample.T  # (num_bits, num_reads)
            sampleset = np.unique(sampleset, axis=1)

            # get the index of the `check_num_obj` smallest qubo_obj values
            qubo_obj_array = np.diag(sampleset.T @ Q @ sampleset)
            real_check_num_obj = min(len(qubo_obj_array), check_num_obj)
            idx_min_qubo_obj_array = np.argpartition(qubo_obj_array, real_check_num_obj - 1)[:real_check_num_obj]

            # get minimum obj and qubo_obj function value
            obj_array = np.zeros(real_check_num_obj, dtype=float)
            for i in range(real_check_num_obj):
                if flip:
                    tmp_resp = self.problem.fwd_modeling(
                        flip_bits(current_bin_model_params, sampleset[:, idx_min_qubo_obj_array[i]])
                    )
                else:
                    tmp_resp = self.problem.fwd_modeling(sampleset[:, idx_min_qubo_obj_array[i]])
                obj_array[i] = self.problem.obj_func(tmp_resp, self.problem.obs_resp)
            if rng.choice([True, False], p=[prob_random_obj, 1 - prob_random_obj]):
                obj = rng.choice(obj_array)
            else:
                obj = np.min(obj_array)
            idx_min_obj_value = int(idx_min_qubo_obj_array[obj_array == obj][0])
            qubo_obj = qubo_obj_array[idx_min_obj_value]

            # get temporary binary model parameters
            q = sampleset.astype(float, copy=False)[:, idx_min_obj_value]
            if flip:
                tmp_bin_model_params = np.floor(flip_bits(current_bin_model_params, q)).astype(int, copy=False)
            else:
                tmp_bin_model_params = np.floor(q).astype(int, copy=False)

            if verbose:
                self._show_verbose_iter(iter_num + 1, tmp_bin_model_params, qubo_obj, obj)

            # append to history
            history["bin_model_params"].append(np.copy(tmp_bin_model_params))
            history["qubo_obj"].append(qubo_obj)
            history["obj"].append(obj)

            # stopping/saving criteria
            if obj <= obj_tol:
                if verbose:
                    print(f"* Stopping b/c obj <= {obj_tol}")
                break
            if repeat_solution_stop_condition and np.array_equal(
                history["bin_model_params"][iter_num], tmp_bin_model_params
            ):
                if verbose:
                    print("* Stopping b/c of repeated solutions")
                break

            # update current_bin_model_params
            current_bin_model_params = np.copy(tmp_bin_model_params)

        # get the index of the smallest objective value from the history
        tmp_idx = int(np.argmin(history["obj"]))

        return {
            "bin_model_params": history["bin_model_params"][tmp_idx],
            "qubo_obj": history["qubo_obj"][tmp_idx],
            "obj": history["obj"][tmp_idx],
            "history": history,
        }
