from time import time
from py_expression_eval import Parser
import mexpress as mx
import numpy as np
from math import sin, cos, tan
from sympy import sympify


def timed_parse(name, func, s, n_runs=1000):
    res = None
    t0 = time()
    for _ in range(n_runs):
        res = func(s)
    elapsed = time() - t0
    print(f"parsing {n_runs} times {name:18} took {elapsed:.10f}")
    return res


def timed_eval(name, func, args, n_runs=1000, **kwargs):
    res = None
    t0 = time()
    for _ in range(n_runs):
        if len(args) > 0:
            res = func(args, **kwargs)
        else:
            res = func(**kwargs)
    elapsed = time() - t0
    print(f"evaluating {n_runs} times {name:18} took {elapsed:.10f}")
    return res


def main():

    s = "sin(x) + tan(y) / (sin(z)**2 + cos(z)**2)"

    mexpr = timed_parse("mexpress", mx.parse_f64, s)
    parser = Parser()
    axia = timed_parse("py_expression_eval", parser.parse, s)
    symp_expr = timed_parse("sympy", sympify, s)

    timed_eval("mexpress", mexpr, np.array((1.0, 2.0, 3.0), dtype=np.float64))
    timed_eval("py_expression_eval", axia.evaluate, {"x": 1.0, "y": 2.0, "z": 3.0})
    timed_eval("sympy", symp_expr.evalf, [], subs={"x": 1.0, "y": 2.0, "z": 3.0})

    def plain(args):
        return sin(args[0]) + tan(args[1]) / (sin(args[2]) ** 2 + cos(args[2]) ** 2)

    timed_eval("plain", plain, (1.0, 2.0, 3.0))


if __name__ == "__main__":
    main()
