from abc import ABC, abstractmethod
from collections import ChainMap
from typing import Any, MutableMapping, Optional

import dimod
import numpy as np


class BaseSolver(ABC):
    def __init__(
        self,
        problem,
        sampler: Optional[dimod.Sampler] = None,
        sampling_params: MutableMapping[str, Any] = {},
    ) -> None:

        self._check_sampling_params(sampling_params)

        self.problem = problem
        self.sampler = sampler
        self.sampling_params = dict(ChainMap(sampling_params, self.default_sampling_params))

    @property
    def default_sampling_params(self):
        return {}

    @staticmethod
    def _check_sampler(sampler):
        if not hasattr(sampler, "sample_qubo"):
            raise AttributeError(f"Sampler must have the `sampler_qubo` method.")

    @staticmethod
    def _check_sampling_params(sampling_params):
        if not isinstance(sampling_params, MutableMapping):
            raise TypeError(
                (
                    "`sampling_params` must be an instance of `MutableMapping`, "
                    f"not an instance of {type(sampling_params)}"
                )
            )


class BaseDirectSolver(BaseSolver):
    @abstractmethod
    def solve(
        self,
        sampler: Optional[dimod.Sampler] = None,
        sampling_params: MutableMapping[str, Any] = {},
    ):
        pass


class BaseIterativeSolver(BaseSolver):
    @abstractmethod
    def solve(
        self,
        initial_guess: np.ndarray,
        sampler: Optional[dimod.Sampler] = None,
        sampling_params: MutableMapping[str, Any] = {},
        iter_params: MutableMapping[str, Any] = {},
    ):
        pass
