from time import time

import numpy as np
from numpy.core.fromnumeric import argmin
from numpy.lib.function_base import median
from scipy.optimize import minimize

import mexpress


def _run_method(f, method, jac, hess, n_runs=5):
    elapseds = []
    xes = []
    for _ in range(n_runs):
        s = time()
        x = minimize(f, [-1.0, -1.0], method=method, jac=jac, hess=hess)
        elapsed = time() - s
        elapseds.append(elapsed)
        xes.append(x)
    med_idx = np.argsort(elapseds)[n_runs // 2]
    elapsed = elapseds[med_idx]
    x = xes[med_idx]
    min_len = 12

    return (
        f"{method:{min_len}}",
        f"({x['x'][0]:8.5f}, {x['x'][1]:8.5f})",
        f"{elapsed:.6f} sec",
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
        ("BFGS", lambda f: f.grad, lambda f: None),
        ("BFGS", lambda f: None, lambda f: None),
        ("L-BFGS-B", lambda f: f.grad, lambda f: None),
        ("L-BFGS-B", lambda f: None, lambda f: None),
        ("Nelder-Mead", lambda f: None, lambda f: None),
        ("SLSQP", lambda f: f.grad, lambda f: None),
        ("dogleg", lambda f: f.grad, lambda f: f.hess),
    ]

    for m in methods:
        res = _run_method(f, m[0], m[1](f), m[2](f))
        print("   ".join(str(r) for r in res))


if __name__ == "__main__":
    a = 1
    b = 100
    rosen = f"({a}-x) ** 2 + {b}*(y-x ** 2) ** 2 "
    main(rosen)
