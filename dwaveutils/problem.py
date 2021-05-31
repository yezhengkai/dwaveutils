from abc import ABC
from collections import ChainMap
from typing import Any, MutableMapping


class BaseProblem(ABC):
    def __init__(
        self, problem_params: MutableMapping[str, Any] = {}, qubo_params: MutableMapping[str, Any] = {}
    ) -> None:

        self.problem_params = dict(ChainMap(problem_params, self.default_problem_params))
        self.qubo_params = dict(ChainMap(qubo_params, self.default_qubo_params))

    @property
    def default_problem_params(self):
        return {}

    @property
    def default_qubo_params(self):
        return {}

    @property
    def required_problem_params(self):
        return []

    @property
    def required_qubo_params(self):
        return []

    def _check_problem_params(self, problem_params):
        if not isinstance(problem_params, MutableMapping):
            raise TypeError(
                f"`problem_params` must be an instance of `MutableMapping`, not an instance of {type(problem_params)}"
            )
        for param in self.required_problem_params:
            if param not in problem_params:
                raise KeyError(f"The required `problem_params` is {self.required_problem_params}")

    def _check_qubo_params(self, qubo_params):
        if not isinstance(qubo_params, MutableMapping):
            raise TypeError(
                f"`qubo_params` must be an instance of `MutableMapping`, not an instance of {type(qubo_params)}"
            )
        for param in self.required_qubo_params:
            if param not in qubo_params:
                raise KeyError(f"The required `qubo_params` is {self.required_qubo_params}")
