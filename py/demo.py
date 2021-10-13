from time import time

import numpy as np
from numpy.core.fromnumeric import argmin
from numpy.lib.function_base import median
from scipy.optimize import minimize

import mexpress


def _run_method(f, method, jac, hess, n_runs=3):
    elapseds = []
    xes = []
    for _ in range(n_runs):
        s = time()
        x = minimize(f, [-1.0] * f.n_vars, method=method, jac=jac, hess=hess, options={"maxiter": 5000})
        elapsed = time() - s
        elapseds.append(elapsed)
        xes.append(x)
    med_idx = np.argsort(elapseds)[n_runs // 2]
    elapsed = elapseds[med_idx]
    x = xes[med_idx]
    min_len = 12
    x_ = ", ".join([f"{xi:8.5f}" for xi in x["x"]])
    return (
        f"{method:{min_len}}",
        f"({x_})",
        f"{elapsed:.6f} sec",
        f"jac {str(jac is not None):5}",
        f"hess {str(hess is not None):5}",
    )


def main(func_str):
    
    methods = [
        ("CG", lambda f: f.grad, lambda _: None),
        ("CG", lambda _: None, lambda _: None),
        ("Newton-CG", lambda f: f.grad, lambda f: f.hess),
        ("Newton-CG", lambda f: f.grad, lambda _: None),
        ("trust-krylov", lambda f: f.grad, lambda f: f.hess),  # needs Hessian
        ("trust-ncg", lambda f: f.grad, lambda f: f.hess),  # needs Hessian
        ("trust-exact", lambda f: f.grad, lambda f: f.hess),  # needs Hessian
        ("BFGS", lambda f: f.grad, lambda _: None),
        ("BFGS", lambda _: None, lambda _: None),
        ("L-BFGS-B", lambda f: f.grad, lambda _: None),
        ("L-BFGS-B", lambda _: None, lambda _: None),
        ("Nelder-Mead", lambda _: None, lambda _: None),
        ("SLSQP", lambda f: f.grad, lambda _: None),
        ("dogleg", lambda f: f.grad, lambda f: f.hess),
    ]

    res = []

    for m in methods:
        f = mexpress.parse(func_str)
        res = _run_method(f, m[0], m[1](f), m[2](f))
        print("   ".join(str(r) for r in res))


if __name__ == "__main__":
    a = 1
    b = 100
    func = f"({a}-x) ** 2 + {b}*(y-x ** 2) ** 2 + (z - 7) ** 2 + (ρ + 5) ** 2"
    main(func)
