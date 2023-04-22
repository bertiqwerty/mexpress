import numpy as np

from .mexpress import native_parse_f64, native_parse_f32


def _transform_x(x):
    x = np.asarray(x)
    # since we want to have both options f(*x) and f(x) we need to check the dimension
    return x if x.ndim == 1 else x.squeeze()


class Mexpress:
    def __init__(self, interfex, dtype) -> None:
        self.interfex = interfex
        self._grad = None
        self._hess = None
        self.dtype = dtype
        self.n_vars = self.interfex.n_vars()

    def _make_grad(self):
        if self._grad is None:
            self._grad = [
                self.interfex.partial(i) for i in range(self.interfex.n_vars())
            ]
        return self._grad

    def __call__(self, *x):
        return self.interfex(_transform_x(x))

    def partial(self, i: int) -> "Mexpress":
        return Mexpress(self.interfex.partial(i), self.dtype)

    def grad(self, *x) -> np.ndarray:
        grad_ = self._make_grad()
        x = _transform_x(x)
        return np.array([di(x) for di in grad_], dtype=self.dtype)

    def hess(self, *x) -> np.ndarray:
        grad_ = self._make_grad()
        if self._hess is None:
            self._hess = [
                [grad_i.partial(c) for c in range(r, self.interfex.n_vars())]
                for r, grad_i in enumerate(grad_)
            ]
        x = _transform_x(x)
        hess = np.zeros((self.n_vars, self.n_vars), dtype=self.dtype)
        for r in range(self.n_vars):
            for c in range(r, self.n_vars):
                hess[r, c] = self._hess[r][c - r](x)
            for c in range(0, r):
                hess[r, c] = hess[c, r]
        return hess

    def __str__(self) -> str:
        return self.interfex.unparse()


def parse_f64(s: str) -> Mexpress:
    return Mexpress(native_parse_f64(s.replace("**", "^")), dtype=np.float64)


def parse_f32(s: str) -> Mexpress:
    return Mexpress(native_parse_f32(s.replace("**", "^")), dtype=np.float32)
