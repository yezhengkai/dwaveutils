import warnings
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Tuple, Union

import numpy as np

from .utils import QUBO, flip_bits, fwd_modeling, residual_sum_squares


# TODO: refactor bl_lstsq's BaseSolver._assign
class BaseSolver(metaclass=ABCMeta):
    def __init__(
        self,
        fwd_model_func: Callable[[np.ndarray], np.ndarray],
        obs_resp: np.ndarray,
        low_high: Union[Tuple[Union[int, float], Union[int, float]], List[Union[int, float]]],
        params_inv2fwd_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
        resp_all2meas_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
        sampler=None,
        sampling_params=None,
    ) -> None:

        if sampler is not None:
            self._check_sampler(sampler)

        super().__init__()
        self.fwd_model_func = fwd_model_func
        self.obs_resp = obs_resp
        self.low_high = low_high
        self.params_inv2fwd_func = params_inv2fwd_func
        self.resp_all2meas_func = resp_all2meas_func
        self.sampler = sampler
        self.sampling_params = sampling_params
        self.qubo = QUBO(
            fwd_model_func,
            obs_resp,
            low_high,
            params_inv2fwd_func=params_inv2fwd_func,
            resp_all2meas_func=resp_all2meas_func,
        )

    def get_qubo(self, bin_params, return_matrix=False):
        return self.qubo.get(bin_params, return_matrix=return_matrix)

    def _check_sampler(self, sampler):
        if not hasattr(sampler, "sample_qubo"):
            raise AttributeError(f"Sampler must have the `sampler_qubo` method.")

    def _assign(self, params, attr, default_value=None, check_func=None):
        if params is None and getattr(self, attr, None) is None:
            if default_value is not None:
                setattr(self, attr, default_value)
                warnings.warn(f"Use default `{attr}`: {default_value}")
            else:
                raise ValueError(f"Please enter a `{attr}`.")
        elif params is None and getattr(self, attr, None) is not None:
            pass
        else:
            if check_func is not None and callable(check_func):
                check_func(params)
            if default_value is not None and isinstance(default_value, dict) and isinstance(params, dict):
                default_value.update(params)
                setattr(self, attr, default_value)
            else:
                setattr(self, attr, params)

    @abstractmethod
    def solve(self):
        pass


# TODO: replace dwave.inverse.base.oneIterDwaveLinearnl with dwave.inverse.solver.Solver instance
# TODO: refactor the default values of DirectSolver and IterativeSolver of bl_lstsq
class Solver(BaseSolver):

    default_iter_params = {
        "num_iter": 20,
        "rss_tol": 0,
        "check_num_nonlinear_obj": 48,
        "flip": True,
        "repeat_solution_stop_condition": False,
        "verbose": False,
    }

    def __init__(
        self,
        fwd_model_func: Callable[[np.ndarray], np.ndarray],
        obs_resp: np.ndarray,
        low_high: Union[Tuple[Union[int, float], Union[int, float]], List[Union[int, float]]],
        params_inv2fwd_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
        resp_all2meas_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
        sampler=None,
        sampling_params=None,
        iter_params=None,
    ) -> None:
        super().__init__(
            fwd_model_func,
            obs_resp,
            low_high,
            params_inv2fwd_func=params_inv2fwd_func,
            resp_all2meas_func=resp_all2meas_func,
            sampler=sampler,
            sampling_params=sampling_params,
        )
        self.iter_params = iter_params

    def solve(self, initial_bin_model_params, sampler=None, sampling_params=None, iter_params=None):
        self._assign(sampler, "sampler")
        self._assign(sampling_params, "sampling_params")
        self._assign(
            iter_params, "iter_params", default_value=self.default_iter_params, check_func=self._check_iter_params
        )

        current_bin_model_params = np.copy(initial_bin_model_params)
        num_iter = self.iter_params["num_iter"]
        rss_tol = self.iter_params["rss_tol"]
        check_num_nonlinear_obj = self.iter_params["check_num_nonlinear_obj"]
        flip = self.iter_params["flip"]
        repeat_solution_stop_condition = self.iter_params["repeat_solution_stop_condition"]
        verbose = self.iter_params["verbose"]
        history = {
            "bin_model_params": [initial_bin_model_params],
            "linear_obj": [
                initial_bin_model_params.T
                @ self.get_qubo(initial_bin_model_params, return_matrix=True)
                @ initial_bin_model_params
            ],
            "nonlinear_obj": [
                residual_sum_squares(
                    fwd_modeling(
                        self.fwd_model_func,
                        initial_bin_model_params,
                        low_high=self.low_high,
                        params_inv2fwd_func=self.params_inv2fwd_func,
                        resp_all2meas_func=self.resp_all2meas_func,
                    ),
                    self.obs_resp,
                )
            ],
        }

        if verbose:
            self._show_verbose_iter(
                0, history["bin_model_params"][0], history["linear_obj"][0], history["nonlinear_obj"][0]
            )
        for iter_num in range(num_iter):
            # get qubo
            Q = self.get_qubo(current_bin_model_params, return_matrix=True)

            # get sampleset
            sampleset = self.sampler.sample_qubo(Q.toarray(), **self.sampling_params)
            sampleset = sampleset.record.sample.T  # [num_bits, num_reads]

            # get minimum num_nonlinear solutions
            sampleset = np.unique(sampleset, axis=1)
            norm = np.diag(sampleset.T @ Q @ sampleset)
            rt = np.copy(norm)
            nonlinear_values = np.zeros(check_num_nonlinear_obj, dtype=int)
            maxval = np.max(rt)
            for i in range(check_num_nonlinear_obj):
                minidx = np.argmin(rt)
                nonlinear_values[i] = minidx
                rt[minidx] = maxval

            # get minimum nonlinear value
            nonlinear_obj = np.zeros(check_num_nonlinear_obj, dtype=float)
            for i in range(check_num_nonlinear_obj):
                # get non-linear objective function value
                if flip:
                    tmp_resp = fwd_modeling(
                        self.fwd_model_func,
                        flip_bits(current_bin_model_params, sampleset[:, nonlinear_values[i]]),
                        low_high=self.low_high,
                        params_inv2fwd_func=self.params_inv2fwd_func,
                        resp_all2meas_func=self.resp_all2meas_func,
                    )
                else:
                    tmp_resp = fwd_modeling(
                        self.fwd_model_func,
                        sampleset[:, nonlinear_values[i]],
                        low_high=self.low_high,
                        params_inv2fwd_func=self.params_inv2fwd_func,
                        resp_all2meas_func=self.resp_all2meas_func,
                    )
                nonlinear_obj[i] = residual_sum_squares(tmp_resp, self.obs_resp)

            min_value = np.min(nonlinear_obj)
            idx_min_value = nonlinear_values[nonlinear_obj == min_value][0]

            q = sampleset.astype(float, copy=False)[:, idx_min_value]
            linear_obj = norm[idx_min_value]
            nonlinear_obj = min_value

            if flip:
                tmp_bin_model_params = np.floor(flip_bits(current_bin_model_params, q)).astype(int, copy=False)
            else:
                tmp_bin_model_params = np.floor(q).astype(int, copy=False)

            if verbose:
                self._show_verbose_iter(iter_num + 1, tmp_bin_model_params, linear_obj, nonlinear_obj)

            # append to history
            history["bin_model_params"].append(np.copy(tmp_bin_model_params))
            history["linear_obj"].append(linear_obj)
            history["nonlinear_obj"].append(nonlinear_obj)

            # stopping/saving criteria
            if nonlinear_obj <= rss_tol:
                if verbose:
                    print(f"* Stopping b/c nonlinear_obj <= {rss_tol}")
                break

            if repeat_solution_stop_condition and np.array_equal(
                history["bin_model_params"][iter_num], tmp_bin_model_params
            ):
                if verbose:
                    print("* Stopping b/c of repeated solutions")
                break

            # update current_bin_model_params
            current_bin_model_params = np.copy(tmp_bin_model_params)

        # get the index of the smallest nonlinear_obj from the history
        tmp_idx = np.argmin(history["nonlinear_obj"])

        return {
            "bin_model_params": history["bin_model_params"][tmp_idx],
            "linear_obj": history["linear_obj"][tmp_idx],
            "nonlinear_obj": history["nonlinear_obj"][tmp_idx],
            "history": history,
        }

    def _check_iter_params(self, iter_params):
        if not isinstance(iter_params, dict):
            raise TypeError(f"iter_params must be a dict, not {type(iter_params)}")
        if "num_iter" in iter_params:
            if iter_params["num_iter"] <= 0:
                iter_params["num_iter"] = self.default_iter_params["num_iter"]
                warnings.warn(f"Set num_iter = {iter_params['num_iter']}")

    def _show_verbose_iter(self, iter_num, bin_model_params, linear_obj, nonlinear_obj):
        print(f"Iteration: {iter_num}")
        print(f"  - bin_model_params = {bin_model_params}")
        print(f"  - linear_obj    = {linear_obj:.8e}")
        print(f"  - nonlinear_obj = {nonlinear_obj:.8E}")
