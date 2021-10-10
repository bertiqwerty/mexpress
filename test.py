import mexpress as mx
import numpy as np


def _assert_float_eq(x, y):
    assert np.linalg.norm(x - y) < 1e-12


def test_eval():
    expr = mx.parse("x^2 + 7")
    _assert_float_eq(expr(2), 11)

    expr = mx.parse("x^2 + 7")
    _assert_float_eq(expr(2), 11)

    expr = mx.parse("Δ + x^2 + 7")
    _assert_float_eq(expr(2, 23), 34)

    expr = mx.parse("sin(x)^2 + 1/y")
    ref = np.sin(2) ** 2 + 1 / 4
    _assert_float_eq(expr(2, 4), ref)


def test_grad():
    expr = mx.parse("x + y + z")
    _assert_float_eq(expr.grad(1, 2, 3), [1, 1, 1])
    _assert_float_eq(expr.grad(np.random.random(3)), [1, 1, 1])
    _assert_float_eq(expr.grad(np.random.random(3)), [1, 1, 1])

    expr = mx.parse("2*x + y^2 + cos(z) + sin(Δ)")
    ref = lambda _, y, z, Δ: (2, 2 * y, -np.sin(z), np.cos(Δ))
    x = np.random.random(4)
    _assert_float_eq(expr.grad(*x), ref(*x))


def test_hess():

    expr = mx.parse("2*x * y**2 + cos(z) + sin(Δ)")
    res = expr.hess(2, 3, 4, 5)
    ref = np.array(
        [
            [0, 4 * 3, 0, 0],
            [4 * 3, 2 * 2 * 2, 0, 0],
            [0, 0, -np.cos(4), 0],
            [0, 0, 0, -np.sin(5)],
        ],
        dtype=np.float64,
    )
    _assert_float_eq(res, ref)


if __name__ == "__main__":
    test_grad()
    test_hess()
