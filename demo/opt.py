from time import time
import numpy as np
from scipy.optimize import minimize

import mexpress


def _run_method(f, method, jac, hess, n_runs=100, n_it_max=5000):
    elapseds = []
    fails = []
    n_its = []
    np.random.seed(0)
    for _ in range(n_runs):
        x0 = np.random.random(f.n_vars) * 20 - 10
        s = time()
        x = minimize(
            f,
            x0=x0,
            method=method,
            jac=jac,
            hess=hess,
            options={"maxiter": n_it_max},
        )
        elapsed = time() - s
        elapseds.append(elapsed)
        fails.append(not x["success"])
        if "nit" in x:
            n_its.append(x["nit"])
        else:
            n_its.append(-1)

    med_sec_idx = np.argsort(elapseds)[n_runs // 2]
    elapsed = elapseds[med_sec_idx]

    med_it_idx = np.argsort(n_its)[n_runs // 2]

    min_len = 12
    suc = f"{np.sum(np.array(fails)):3d}"
    n_it = f"{n_its[med_it_idx]:3d}"
    return (
        f"{method:{min_len}}",
        f"#fails {suc}",
        f"#it {n_it}",
        f"{elapsed:.10f} sec",
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
        ("TNC", lambda f: f.grad, lambda _: None),
        ("TNC", lambda _: None, lambda _: None),
        ("COBYLA", lambda _: None, lambda _: None),
        ("POWELL", lambda _: None, lambda _: None),
    ]

    res = []

    for m in methods:
        f = mexpress.parse_f64(func_str)
        res = _run_method(f, m[0], m[1](f), m[2](f))
        print("   ".join(str(r) for r in res))


if __name__ == "__main__":
    a = 1
    b = 100
    func = f"({a}-x) ** 2 + {b}*(y-x ** 2) ** 2 + (z - 7) ** 2 + (ρ + 5) ** 2"
    main(func)
