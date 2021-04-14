# References:
# https://github.com/sygreer/QuantumAnnealingInversion.jl/blob/main/src/HelperFunctions.jl
from numpy import ndarray


def checkObj(h0: ndarray, hhat: ndarray) -> float:
    """Checks the full objective function.

    Parameters
    ----------
    h0 : numpy.ndarray
        forward modeled head at all l
    hhat : numpy.ndarray
        measured head at all l

    Returns
    -------
    sumval : float
        value of objective function for the input model
    """
    sumval = 0
    for i in range(len(hhat)):
        sumval += (h0[i] - hhat[i]) ** 2
    return sumval
