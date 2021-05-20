import numpy as np


# TODO: replace dwave.inverse.base.kinn with dwave.inverse.interpolate.nn_1d_interpolate
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


# TODO: replace dwave.inverse.base.twodinterp with dwave.inverse.interpolate.nn_2d_interpolate
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
