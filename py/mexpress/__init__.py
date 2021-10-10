from typing import List
import numpy as np
from .mexpress import native_parse, FlatEx


class Mexpress:
    def __init__(self, flatex: FlatEx) -> None:
        self.flatex = flatex
        self._grad = None
        self._hess = None

    def _make_grad(self) -> List[FlatEx]:
        if self._grad is None:
            self._grad = [self.flatex.partial(i) for i in range(self.flatex.n_vars())]
        return self._grad

    def __call__(self, *x):
        return self.flatex(np.array(x, dtype=np.float64))

    def partial(self, i: int) -> "Mexpress":
        return Mexpress(self.flatex.partial(i))

    def grad(self, *x):
        grad_ = self._make_grad()
        x = np.array(x, dtype=np.float64)
        return np.array([di(x) for di in grad_], dtype=np.float64)

    def hess(self, *x):
        grad_ = self._make_grad()
        if self._hess is None:
            self._hess = [
                [grad_i.partial(i) for i in range(self.flatex.n_vars())] for grad_i in grad_
            ]
        x = np.array(x, dtype=np.float64)
        return np.array([[hess_ij(x) for hess_ij in hess_i] for hess_i in self._hess])

    def __str__(self):
        return self.flatex.unparse()


def parse(s: str) -> Mexpress:
    return Mexpress(native_parse(s.replace("**", "^")))
