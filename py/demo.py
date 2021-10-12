from pprint import pprint
from time import time

import numpy as np
from scipy.optimize import minimize

import mexpress


def _run_method(f, method, jac, hess):
    s = time()
    x = minimize(f, [0.0, 0.0], method=method, jac=jac, hess=hess)
    elapsed = time() - s
    min_len = 12

    return (
        f"{method:{min_len}}",
        f"({x['x'][0]:.5f}, {x['x'][1]:.5f})",
        f"{elapsed:.3f} secs",
        f"jac {str(jac is not None):5}",
        f"hess {str(hess is not None):5}",
    )


def main(func_str):
    f = mexpress.parse(func_str)
    res = []

    methods = [
        ("CG", lambda f: f.grad, lambda f: None),
        ("CG", lambda f: None, lambda f: None),
        ("Newton-CG", lambda f: f.grad, lambda f: f.hess),
        ("Newton-CG", lambda f: f.grad, lambda f: None),
        ("trust-krylov", lambda f: f.grad, lambda f: f.hess),  # needs Hessian
        ("trust-ncg", lambda f: f.grad, lambda f: f.hess),  # needs Hessian
        ("trust-exact", lambda f: f.grad, lambda f: f.hess),  # needs Hessian
    ]

    for m in methods:
        res = _run_method(f, m[0], m[1](f), m[2](f))
        print("   ".join(str(r) for r in res))


if __name__ == "__main__":
    a = 1
    b = 100
    rosen = f"({a}-x) ** 2 + {b}*(y-x ** 2) ** 2 "
    main(rosen)
