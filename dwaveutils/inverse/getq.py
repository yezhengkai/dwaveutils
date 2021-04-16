# References:
# https://github.com/sygreer/QuantumAnnealingInversion.jl/blob/main/src/GetQ.jl
from copy import deepcopy
from typing import Callable

import numpy as np

from ..utils import Binary2Float
from .base import geth0, kinn


def qubo(hhat: np.ndarray, F: np.ndarray, h0: np.ndarray) -> np.ndarray:
    """Get the qubo.

    Parameters
    ----------
    hhat : numpy.ndarray
        measured values of head
    F : numpy.ndarray
        F matrix (operator for flipping bits)
    h0 : numpy.ndarray
        initial head

    Returns
    -------
    numpy.ndarray
        qubo matrix
    """
    Q = F.T @ F + np.diag(2 * F.T @ (h0 - hhat))
    Q2 = np.zeros(Q.shape)

    for i in range(Q.shape[0]):
        Q2[i, i] = Q[i, i]
        for j in range(i):
            Q2[i, j] = Q[i, j] + Q[j, i]
    return Q2.T


def getF(
    forwardModel: Callable[[np.ndarray], np.ndarray],
    k0: np.ndarray,
    klow: float,
    khigh: float,
    inum: int,
    lvals: int = None,
) -> np.ndarray:
    """Get the F matrix LxJ matrix.

    Parameters
    ----------
    forwardModel : Callable object
        forward modeling operator
    k0 : numpy.ndarray
        initial head
    klow : float
        low head value
    khigh : float
        high head value
    inum : int
        number of times i grid is within j grid
    lvals : int, optional
        i index of known l values, by default None

    Returns
    -------
    numpy.ndarray
        F matrix used in qubo function
    """
    # outputs the known values of head
    def h(k0, lvals):
        kinnn = kinn(k0, inum)
        hi = forwardModel(kinnn)
        if lvals is None:
            return hi
        else:
            return hi[lvals]

    hl = h(k0, lvals)
    F = np.zeros((len(hl), len(k0)))
    for j in range(len(k0)):
        k1 = deepcopy(k0)
        if k0[j] == klow:
            k1[j] = khigh
        else:
            k1[j] = klow
        F[:, j] = h(k1, lvals) - hl
    return F


def getQ(
    forwardModel: Callable[[np.ndarray], np.ndarray],
    initGuess: np.ndarray,
    kl: float,
    kh: float,
    inum: int,
    hhat: np.ndarray,
    lvals: int = None,
) -> np.ndarray:
    """Get standard Q value.

    Parameters
    ----------
    forwardModel : Callable
        forward modeling operator
    initGuess : numpy.ndarray
        guess for permeability
    kl : float
        low permeability
    kh : float
        high permeability
    inum : int
        number of times i grid is within j grid
    hhat : numpy.ndarray
        measured head values
    lvals : int, optional
        location of head measurements, by default None

    Returns
    -------
    numpy.ndarray
        Q matrix
    """
    print("Calculating Q...")
    h0 = geth0(forwardModel, initGuess, kl, kh, inum, lvals)
    kjGuess = Binary2Float.to_two_value(initGuess, kl, kh)  # convert to actual perm
    F = getF(forwardModel, kjGuess, kl, kh, inum, lvals)
    Q = qubo(hhat, F, h0)  # get Q matrix
    print("Done.", end="\n\n")
    return Q
