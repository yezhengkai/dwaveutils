# References:
# https://github.com/sygreer/QuantumAnnealingInversion.jl/blob/main/src/QuantumAnnealingInversion.jl
import warnings
from collections import defaultdict
from typing import Any, Callable, Tuple, Union

import numpy as np
from dimod.meta import SamplerABCMeta
from dwave.system import DWaveSampler, EmbeddingComposite
from neal import SimulatedAnnealingSampler

from ..utils import Binary2Float
from .helper import checkObj


def geth0(
    forwardModel: Callable[[np.ndarray], np.ndarray],
    soln: np.ndarray,
    kl: float,
    kh: float,
    inum: int,
    lvals: Union[int, None] = None,
) -> np.ndarray:
    """Get head measurements.

    Parameters
    ----------
    forwardModel : Callable
        forward modeling operator
    soln : np.ndarray
        binary value of permeability
    kl : float
        low permeability
    kh : float
        high permeability
    inum : int
        number of times i grid is within j grid
    lvals : int or None, optional
        index of head measurements, by default None

    Returns
    -------
    np.ndarray
        measured head values
    """
    kjGuess = Binary2Float.to_two_value(soln, kl, kh)  # convert to actual perm
    kiGuess = kinn(kjGuess, inum)  # interpolate
    hGuess = forwardModel(kiGuess)  # forward model to get head
    if lvals is None:
        return hGuess
    else:
        return hGuess[lvals]  # get initial head value


def kinn(kj: np.ndarray, inum: int) -> np.ndarray:
    """Nearest neighbor interpolation.

    Parameters
    ----------
    kj : numpy.ndarray
        k values on j grid
    inum : int
        number of times i grid is within j grid

    Returns
    -------
    ki : numpy.ndarray
        nearest neighbor interpolated k values on i grid
    """
    ki = np.ones(inum * len(kj))  # kj nn interp to grid i
    for i in range(len(ki)):  # convert kj to ki
        val = int(np.floor(i / inum))
        ki[i] = kj[val]
    return ki


def twodinterp(arr: np.ndarray, intval: int) -> np.ndarray:
    """Interpolate in 2D array.

    Parameters
    ----------
    arr : numpy.ndarray
        An array to interpolation
    intval : int
        number of times i grid is within j grid

    Returns
    -------
    retarr : numpy.ndarray
        Interpolated array
    """
    if arr.ndim > 2:
        raise ValueError(
            "Error when checking input: expected arr to have dimensions"
            + f" 1 or 2 but got arr with dimensions {arr.ndim}"
        )
    elif arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    retarr = np.empty(np.multiply(arr.shape, intval))
    for i in range(arr.shape[0]):
        for j in range(intval):
            retarr[i * intval + j, :] = kinn(arr[i, :], intval)
    return retarr


def flipBits(k0: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Update the solution.

    Parameters
    ----------
    k0 : numpy.ndarray
        initial guess
    q : numpy.ndarray
        q values

    Returns
    -------
    numpy.ndarray
        updated guess
    """
    assert np.isin(k0, [0, 1]).all(), "The elements in the k0 array must be 0 or 1"
    assert np.isin(q, [0, 1]).all(), "The elements in the q array must be 0 or 1"
    return abs(k0 - q)


def getPermutationsDwaveLinearnl(
    forwardModel: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    hhat: np.ndarray,
    initGuess: np.ndarray,
    kl: float,
    kh: float,
    inum: int,
    lvals: Union[int, None],
    flip: bool = True,
    num_reads: int = 10000,
) -> Tuple[np.ndarray, float, float]:
    """Get solutions from the D-Wave.

    Parameters
    ----------
    forwardModel : Callable
        forward modeling operator
    Q : numpy.ndarray
        qubo matrix
    hhat : numpy.ndarray
        h measurements
    initGuess : numpy.ndarray
        initial guess of permeability
    kl : float
        low permeability
    kh : float
        high permeability
    inum : int
        number of times i grid is within j grid
    lvals : int or None
        location of measured head values
    flip : bool, optional
        Whether to flip, by default True
    num_reads : int, optional
        Number of reads, by default 10000

    Returns
    -------
    numpy.ndarray
        all valid permutations
    float
        the linear and non-linear objective function value of that permutation
    float
        the indices of permutations that minimize the non-linear objective function
    """
    ##################################################
    # CHANGE BETWEEN QBSOLV AND DWAVE HERE
    print("Getting D-Wave solutions...")
    kbs = getDwaveAnswer(Q, num_reads)
    # kbs = getDwaveAnswer(Q, mytoken, param_chain=param_chain)
    # kbs = getDwaveAnswerqbsolv(Q, mytoken)      # works without internet
    print("Done.\n")
    ##################################################

    kbs = np.unique(kbs, axis=1)
    norm = np.zeros(kbs.shape[1])

    for i in range(kbs.shape[1]):
        norm[i] = kbs[:, i].T @ Q @ kbs[:, i]

    # get minimum numnl solutions
    rt = np.copy(norm)
    # numnl = 8
    # numnl = 24
    numnl = 48
    nlvals = np.zeros(numnl, dtype=int)
    maxval = np.max(rt)
    for i in range(numnl):
        minidx = np.argmin(rt)
        nlvals[i] = minidx
        rt[minidx] = maxval

    # get min nl val
    nlObj = np.zeros(numnl, dtype=float)
    # nlObj = SharedArray{Float64}(numnl)
    # @sync @distributed for i = 1:numel
    for i in range(numnl):
        # get non-linear objective function value
        if flip:
            h02 = geth0(forwardModel, flipBits(initGuess, kbs[:, nlvals[i]]), kl, kh, inum, lvals)
        else:
            h02 = geth0(forwardModel, kbs[:, nlvals[i]], kl, kh, inum, lvals)
        nlObj[i] = checkObj(h02, hhat)

    minVal = np.min(nlObj)
    # idxMinVal = nlvals[findall(x -> x == minVal, nlObj)[1]]
    idxMinVal = nlvals[nlObj == minVal][0]

    return kbs.astype(float, copy=False)[:, idxMinVal], norm[idxMinVal], minVal


def getPermutationsDwave(forwardModel, Q, hhat, initGuess, kl, kh, inum, lvals, flip=True, num_reads=10000):
    """Get solutions from the D-Wave.

    Parameters
    ----------
    forwardModel : Callable object
        forward modeling operator
    Q : numpy.ndarray
        qubo matrix
    hhat : numpy.ndarray
        h measurements
    initGuess : numpy.ndarray
        initial guess of permeability
    kl : float
        low permeability
    kh : float
        high permeability
    inum : int
        number of times i grid is within j grid
    lvals : int
        location of measured head values
    flip : bool, optional
        Whether to flip, by default True
    num_reads : int, optional
        Number of reads, by default 10000

    Returns
    -------
    numpy.ndarray
        all valid permutations
    float
        the linear objective function value of that permutation
    float
        the non-linear objective function value of that permutation
    int
        the indices of permutations that minimize the non-linear objective function
    """
    ##################################################
    # CHANGE BETWEEN QBSOLV AND DWAVE HERE
    kbs = getDwaveAnswer(Q, num_reads=num_reads)
    # kbs = getDwaveAnswerqbsolv(Q, mytoken)      # works without internet
    ##################################################

    kbs = np.unique(kbs, axis=1)
    norm = np.zeros(kbs.shape[1])
    nlObj = np.zeros(kbs.shape[1])

    for i in range(kbs.shape[1]):
        norm[i] = kbs[:, i].T @ Q @ kbs[:, i]
        # @show sum(kbs[:,i])

        # get non-linear objective function value
        if flip:
            h0 = geth0(forwardModel, flipBits(initGuess, kbs[:, i]), kl, kh, inum, lvals)
        else:
            h0 = geth0(forwardModel, kbs[:, i], kl, kh, inum, lvals)
        nlObj[i] = checkObj(h0, hhat)

    minVal = np.min(nlObj)
    idxMinVal = np.where(nlObj == minVal)[0][0]
    minVallin = np.min(norm)
    idxMinVallin = np.where(norm == minVallin)[0][0]

    if idxMinVal != idxMinVallin:
        print("WARNING: non-linear and linear minimum NOT the same\n")
        print(idxMinVal, idxMinVallin)
        print(kbs[:, idxMinVal] - kbs[:, idxMinVallin])
    # @printf("nlmin w/ corrections: %f; nlmin w/o corrections: %f\n", minVal2, minVal)

    return kbs.astype(float, copy=False), norm, nlObj, idxMinVal


def getDwaveAnswer(
    Q_mat: np.ndarray, num_reads: int = 10000, sampler: Union[str, SamplerABCMeta, Any] = "SA"
) -> np.ndarray:
    """Get the answers from the D-Wave.

    Parameters
    ----------
    Q_mat : numpy.ndarray
        Q matrix
    num_reads : int, optional
        Number of reads, by default 10000
    sampler : str or SamplerABCMeta or Any, optional
        Choose sampler, by default SA

    Returns
    -------
    numpy.ndarray
        kbs solutions, shape is [num_bits, num_reads]
    """
    Q_dict = defaultdict(int)
    for i in range(Q_mat.shape[0]):
        for j in range(Q_mat.shape[1]):
            Q_dict[(i, j)] = Q_mat[i, j]

    # instantiate an object with `sample_qubo` method
    if isinstance(sampler, str) and sampler.upper() == "SA":
        _sampler = SimulatedAnnealingSampler()
    elif isinstance(sampler, str) and sampler.upper() == "QA":
        _sampler = EmbeddingComposite(DWaveSampler())
    elif isinstance(sampler, SamplerABCMeta):
        try:
            _sampler = EmbeddingComposite(sampler())
        except ValueError:
            _sampler = sampler()
    elif hasattr(sampler, "sample_qubo"):
        _sampler = sampler
    else:
        warnings.warn("Wrong sampler, use DWaveSampler instead!")
        _sampler = EmbeddingComposite(DWaveSampler())

    sampleset = _sampler.sample_qubo(Q_dict, num_reads=num_reads)
    kbs = sampleset.record.sample.T  # [num_bits, num_reads]
    return kbs


def oneIterDwaveLinearnl(
    forwardModel: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray,
    initGuess: np.ndarray,
    kl: float,
    kh: float,
    inum: int,
    lvals: Union[int, None],
    hhat: np.ndarray,
    iterNum: int,
    kBin: np.ndarray,
    flip: bool = True,
    num_reads: int = 10000,
) -> Tuple[np.ndarray, float, float]:
    """Get one iteration of the model update.

    Parameters
    ----------
    forwardModel : Callable
        forward modeling operator
    Q : numpy.ndarray
        qubo matrix
    initGuess : numpy.ndarray
        initial guess of solution
    kl : float
        low permeability
    kh : float
        high permeability
    inum : int
        number of times i grid is within j grid
    lvals : int or None
        locations where sensors are
    hhat : numpy.array
        measured head at all l
    iterNum : int
        iteration number
    kBin : numpy.adarray
        binary permeability
    flip : bool, optional
        Whether to flip, by default True
    num_reads : int, optional
        Number of reads, by default 10000

    Returns
    -------
    soln : numpy.ndarray
        updated solution
    objSoln : float
        linear objective function values
    nlObj : float
        non-linear objective function values
    """
    # check all possible solutions
    q, objSoln, nlObj = getPermutationsDwaveLinearnl(
        forwardModel, Q, hhat, initGuess, kl, kh, inum, lvals, flip=flip, num_reads=num_reads
    )

    print("ITERATION NUMBER %i\n" % iterNum)
    if flip:
        soln = np.floor(flipBits(initGuess, q)).astype(int, copy=False)
    else:
        soln = np.floor(q).astype(int, copy=False)

    print(f"     h{iterNum} = {soln}")
    print("     # wrong = %i, lobj = %f, nobj = %.8E" % (sum(abs(kBin - soln)), objSoln, nlObj))

    return soln, objSoln, nlObj


def oneIterDwave(
    forwardModel, Q, initGuess, kl, kh, inum, lvals, hhat, iterNum, kBin, flip=True, num_reads=10000, retSolns=False
):
    """Get one iteration of the model update.

    Parameters
    ----------
    forwardModel : Callable object
        forward modeling operator
    Q : numpy.ndarray
        Q matrix
    initGuess : numpy.ndarray
        initial guess of solution
    kl : float
        k_low
    kh : float
        k_high
    inum : int
        number of times i grid is within j grid
    lvals : int
        locations where sensors are
    hhat : numpy.array
        measured values of head
    iterNum : int
        iteration number
    kBin : numpy.array
        binary permeability
    flip : bool, optional
        Whether to flip, by default True
    num_reads : int, optional
        Number of reads, by default 10000
    retSolns : bool, optional
        Whether to return all solutions, by default False

    Returns
    -------
    soln : numpy.ndarray
        updated solution
    objVal[binVal]: float
        linear objective function values
    nlObjVal[binVal] : float
        non-linear objective function values
    """
    # Q = getQFunction(initGuess, kl, kh, inum, lvals)
    # Q2 = getQ(initGuess, kl, kh, inum, lvals)
    # @show Q2 == Q

    # check all possible solutions
    # permsValue, objVal, binVal = getPermutations(Q)
    #
    permsValue, objVal, nlObjVal, binVal = getPermutationsDwave(
        forwardModel, Q, hhat, initGuess, kl, kh, inum, lvals, flip=flip, num_reads=num_reads
    )

    # get the first value...
    # binVal = binVal[0]

    # display parameters
    # get all q values that minimize
    # q = permsValue[:, binVal]
    # objSoln = q.T @ Q @ q

    # @printf("ITERATION NUMBER %i, lobj_true = %f\n",iterNum, objSoln[1])
    print("ITERATION NUMBER %i\n" % iterNum)
    q = permsValue[:, binVal[0]]
    if flip:
        soln = np.floor(flipBits(initGuess, q)).astype(int, copy=False)
    else:
        soln = np.floor(q).astype(int, copy=False)

    print("     h%i = ", iterNum)
    print(soln)
    print("\n     # wrong = %i, " % sum(abs(kBin - soln)))

    print("lobj = %f" % objVal[binVal])
    print(", nobj = %.8E\n" % nlObjVal[binVal])
    print("\n\n")

    if retSolns:
        if flip:
            allsolns = 0 * permsValue
            for i in range(allsolns.shape[1]):
                # for i = 1:size(allsolns,2)
                allsolns[:, i] = flipBits(permsValue[:, i], initGuess)
            allsolns = np.floor(allsolns).astype(int, copy=False)
        else:
            allsolns = permsValue
        linobjvals = np.zeros(allsolns.shape[1])
        for i in range(allsolns.shape[1]):
            # for i = 1:size(allsolns,2)
            linobjvals[i] = allsolns[:, i].T @ Q @ allsolns[:, i]
        return soln, objVal[binVal], nlObjVal[binVal], allsolns, linobjvals
    else:
        return soln, objVal[binVal], nlObjVal[binVal]
