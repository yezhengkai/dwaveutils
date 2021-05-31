"""Utility functions and classes for dwaveutils module."""
from typing import Any, Union

import numpy as np


class Binary2Float(object):
    @staticmethod
    def to_fixed_point(binary: np.ndarray, bit_value: np.ndarray) -> np.ndarray:
        """Convert a binary array to a floating-point array represented by bit values.

        Parameters
        ----------
        binary : np.ndarray
            Binary array.
        bit_value : np.ndarray
            For constructing fixed-point value.

        Returns
        -------
        float_array : np.ndarray
            Floating-point array.
        """

        # sanity check
        if not isinstance(binary, np.ndarray):
            raise TypeError(f"`binary` must be the instance of np.ndarray, not {type(binary)}")
        if not isinstance(bit_value, np.ndarray):
            raise TypeError(f"`bit_value` must be the instance of np.ndarray, not {type(bit_value)}")

        binary, bit_value = binary.flatten(), bit_value.flatten()
        num_binary_entry = len(binary)
        num_bits = len(bit_value)
        num_x_entry = num_binary_entry // num_bits
        if num_x_entry * num_bits != num_binary_entry:
            raise ValueError("The length of q or bit_value is incorrect.")
        float_array = np.array([bit_value @ binary[i * num_bits : (i + 1) * num_bits] for i in range(num_x_entry)])

        return float_array

    @staticmethod
    def to_two_value(binary: np.ndarray, low_value: Union[int, float], high_value: Union[int, float]) -> np.ndarray:
        """Convert a binary array to a floating-point array represented by two values.

        Parameters
        ----------
        binary : np.ndarray
            Binary array.
        low_value : Union[int, float]
            Low value.
        high_value : Union[int, float]
            High value.

        Returns
        -------
        float_array : np.ndarray
            Floating-point array.
        """
        # sanity check
        if not isinstance(binary, np.ndarray):
            raise TypeError(f"`binary` must be the instance of np.ndarray, not {type(binary)}")
        if not isinstance(low_value, (int, float)):
            raise TypeError(f"`low_value` must be the instance of float or int, not {type(low_value)}")
        if not isinstance(high_value, (int, float)):
            raise TypeError(f"`high_value` must be the instance of float or int, not {type(high_value)}")
        if low_value > high_value:
            low_value, high_value = high_value, low_value
        elif low_value == high_value:
            raise ValueError("`low_value` and `high_value` are the same.")

        float_array = np.array(binary.flatten() * high_value - (binary - 1) * low_value)

        return float_array


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
    return np.linalg.norm(obs - pred)  # equal to sqrt(residual_sum_squares(pred, obs))


def nn_1d_interpolate(arr: np.ndarray, magnification: int) -> np.ndarray:
    """1D nearest neighbor interpolation.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be interpolated.
    magnification : int
        Magnification of array shape.

    Returns
    -------
    interpolated_arr : numpy.ndarray
        Interpolated array.
    """
    if isinstance(arr, np.ndarray):
        raise TypeError(f"`arr` must be a numpy.ndarray, not {type(arr)}")
    if isinstance(magnification, int):
        raise TypeError(f"`magnification` must be a int, not {type(magnification)}")
    if arr.ndim > 1:
        raise ValueError(
            "Error when checking input: expected `arr` to have dimensions"
            + f" 1 but got `arr` with dimensions {arr.ndim}"
        )

    interpolated_arr = np.ones(magnification * len(arr))
    for i in range(len(interpolated_arr)):
        val = int(np.floor(i / magnification))
        interpolated_arr[i] = arr[val]
    return interpolated_arr


def nn_2d_interpolate(arr: np.ndarray, magnification: int) -> np.ndarray:
    """2D nearest neighbor interpolation.

    Parameters
    ----------
    arr : numpy.ndarray
        The array to be interpolated.
    magnification : int
        Magnification of array shape.

    Returns
    -------
    interpolated_arr : numpy.ndarray
        Interpolated array.
    """
    if isinstance(arr, np.ndarray):
        raise TypeError(f"`arr` must be a numpy.ndarray, not {type(arr)}")
    if isinstance(magnification, int):
        raise TypeError(f"`magnification` must be a int, not {type(magnification)}")
    if arr.ndim > 2:
        raise ValueError(
            "Error when checking input: expected arr to have dimensions"
            + f" 1 or 2 but got arr with dimensions {arr.ndim}"
        )
    elif arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    interpolated_arr = np.empty(np.multiply(arr.shape, magnification))
    for i in range(arr.shape[0]):
        for j in range(magnification):
            interpolated_arr[i * magnification + j, :] = nn_1d_interpolate(arr[i, :], magnification)
    return interpolated_arr
