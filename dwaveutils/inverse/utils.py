"""Utility functions and classes for inverse module."""
from collections import defaultdict
from copy import deepcopy
from numbers import Real
from typing import Callable, List, Tuple, TypeVar, Union

import numpy as np
import scipy.sparse as sp

from ..utils import Binary2Float


# TODO: replace dwave.inverse.helper.checkObj with dwave.inverse.utils.residual_sum_squares
def residual_sum_squares(pred: np.ndarray, obs: np.ndarray) -> float:
    """Calculate the residual sum of squares (RSS).

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted dataset.
    obs : numpy.ndarray
        Observed dataset.

    Returns
    -------
    float
        Residual sum of squares.
    """
    return np.sum(np.square((obs - pred)))


def l2_residual(pred: np.ndarray, obs: np.ndarray) -> float:
    return np.linalg.norm(obs - pred)  # == sqrt(residual_sum_squares)


# TODO: replace dwave.inverse.base.geth0 with dwave.inverse.utils.get_obs
def fwd_modeling(
    fwd_model_func: Callable[[np.ndarray], np.ndarray],
    model_params: np.ndarray,
    low_high: Union[Tuple[Union[int, float], Union[int, float]], List[Union[int, float]], None] = None,
    params_inv2fwd_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
    resp_all2meas_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
) -> np.ndarray:
    """Get measurements.

    Parameters
    ----------
    fwd_model_func : Callable
        Forward modeling operator.
    model_params : numpy.ndarray
        The model parameters (inverted mesh) represented by binary or floating point value.
    low_high : tuple, list or None, optional
        Low and high values for converting binary to float, by default None.
    params_inv2fwd_func : Callable or None, optional
        The function converts the parameters in the inverted mesh to the forward mesh, by default None.
    resp_all2meas_func : Callable or None, optional
        The function converts all responses into specific measurement values, by default None.

    Returns
    -------
    resp : numpy.ndarray
        Calculated measurement value.
    """
    if low_high is not None:
        model_params = Binary2Float.to_two_value(model_params, *sorted(low_high))  # convert to actual float

    if params_inv2fwd_func is not None:
        model_params = params_inv2fwd_func(model_params)  # e.g. interpolate mesh

    resp = fwd_model_func(model_params)  # get forward response

    if resp_all2meas_func is None:
        return resp  # indentity map
    else:
        return resp_all2meas_func(resp)  # e.g. sample subset of solutions


# TODO: replace dwave.inverse.base.flipBits with dwave.inverse.utils.flip_bits
def flip_bits(arr: np.ndarray, flip_indicator: np.ndarray) -> np.ndarray:
    """Update the solution.

    Parameters
    ----------
    arr : numpy.ndarray
        The bit array that needs to be flipped.
    flip_indicator : numpy.ndarray
        An array indicating whether the bits need to be flipped.

    Returns
    -------
    numpy.ndarray
        Flipped array.
    """
    assert arr.shape == flip_indicator.shape, "The `arr` and `flip_indicator` arrays must have the same shape"
    assert np.isin(arr, [0, 1]).all(), "The elements in the `arr` array must be 0 or 1"
    assert np.isin(flip_indicator, [0, 1]).all(), "The elements in the `flip_indicator` array must be 0 or 1"
    return abs(arr - flip_indicator)


# TODO: replace dwave.inverse.getq.qubo with dwave.inverse.utils.get_qubo
def get_qubo(
    F: np.ndarray, pred_resp: np.ndarray, obs_resp: np.ndarray, return_matrix: bool = False
) -> Union[defaultdict, sp.dok_matrix]:
    """Get the QUBO dictionary or matrix.

    Get coefficients of a quadratic unconstrained binary optimization (QUBO)
    problem defined by the dictionary.

    Parameters
    ----------
    F : numpy.ndarray
        Flip disturb matrix.
    pred_resp : numpy.ndarray
        Predicted values.
    obs_resp : numpy.ndarray
        Observed values.
    return_matrix : bool
        Whether to return scipy.sparse.dok.dok_matrix.

    Returns
    -------
    defaultdict or scipy.sparse.dok.dok_matrix
        QUBO dictionary or matrix.
    """
    qubo_a = sp.diags(np.diag(F.T @ F) + 2 * F.T @ (pred_resp - obs_resp), format="dok")
    qubo_b = sp.dok_matrix(2 * np.triu(F.T @ F, k=1))

    if return_matrix:
        return qubo_a + qubo_b  # upper triangle dok_matrix
    else:
        Q = defaultdict(int, (qubo_a + qubo_b).items())
        # force the diagonal entries to have values
        for i in range(qubo_a.size):
            Q[(i, i)] += 0.0
        return Q


# TODO: replace dwave.inverse.getq.getF with dwave.inverse.utils.get_flip_disturb_matrix instance
def get_flip_disturb_matrix(
    fwd_model_func: Callable[[np.ndarray], np.ndarray],
    bin_model_params: np.ndarray,
    low_high: Union[Tuple[Union[int, float], Union[int, float]], List[Union[int, float]]],
    params_inv2fwd_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
    resp_all2meas_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
) -> np.ndarray:
    """Get the Flip disturb matrix (F matrix) of shape num_resp x num_param.

    Parameters
    ----------
    fwd_model_func : Callable
        Forward modeling operator.
    bin_model_params : numpy.ndarray
        The parameters (inverted mesh) represented by binary value.
    low_high : tuple, list or None, optional
        Low and high values for converting binary to float, by default None.
    params_inv2fwd_func : Callable or None, optional
        The function converts the parameters in the inverted mesh to the forward mesh, by default None.
    resp_all2meas_func : Callable or None, optional
        The function converts all responses into specific measurement values, by default None.

    Returns
    -------
    numpy.ndarray
        Flip disturb matrix.
    """
    low, high = sorted(low_high)
    resp_ori = fwd_modeling(
        fwd_model_func,
        bin_model_params,
        low_high,
        params_inv2fwd_func=params_inv2fwd_func,
        resp_all2meas_func=resp_all2meas_func,
    )
    F = np.zeros((len(resp_ori), len(bin_model_params)))
    for j in range(len(bin_model_params)):
        flipped_bin_model_params = deepcopy(bin_model_params)
        if bin_model_params[j] == low:
            flipped_bin_model_params[j] = high
        else:
            flipped_bin_model_params[j] = low
        F[:, j] = (
            fwd_modeling(
                fwd_model_func,
                flipped_bin_model_params,
                low_high,
                params_inv2fwd_func=params_inv2fwd_func,
                resp_all2meas_func=resp_all2meas_func,
            )
            - resp_ori
        )
    return F


# TODO: replace dwave.inverse.getq.getQ with dwave.inverse.utils.QUBO instance
class QUBO(object):
    def __init__(
        self,
        fwd_model_func: Callable[[np.ndarray], np.ndarray],
        obs_resp: np.ndarray,
        low_high: Union[Tuple[Union[int, float], Union[int, float]], List[Union[int, float]]],
        params_inv2fwd_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
        resp_all2meas_func: Union[Callable[[np.ndarray], np.ndarray], None] = None,
    ) -> None:
        super().__init__()
        self.fwd_model_func = fwd_model_func
        self.obs_resp = obs_resp
        self.low_high = sorted(low_high)
        self.params_inv2fwd_func = params_inv2fwd_func
        self.resp_all2meas_func = resp_all2meas_func

    def get(self, bin_model_params: np.ndarray, return_matrix: bool = False) -> Union[defaultdict, sp.dok_matrix]:
        pred_resp = fwd_modeling(
            self.fwd_model_func,
            bin_model_params,
            self.low_high,
            params_inv2fwd_func=self.params_inv2fwd_func,
            resp_all2meas_func=self.resp_all2meas_func,
        )
        F = get_flip_disturb_matrix(
            self.fwd_model_func,
            bin_model_params,
            self.low_high,
            params_inv2fwd_func=self.params_inv2fwd_func,
            resp_all2meas_func=self.resp_all2meas_func,
        )
        Q = get_qubo(F, pred_resp, self.obs_resp, return_matrix=return_matrix)
        return Q
