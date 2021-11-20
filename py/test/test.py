import numpy as np

import pytest
import mexpress as mx


def _assert_float_eq(x, y, tol=1e-12):
    assert np.linalg.norm(x - y) < tol


def test_eval():

    with pytest.raises(TypeError):
        mx.parse_f64("")

    expr = mx.parse_f64("x^2 + 7")
    _assert_float_eq(expr(2.0), 11)

    expr = mx.parse_f64("x^2 + 7")
    _assert_float_eq(expr(2.0), 11)

    expr = mx.parse_f64("Δ + x^2 + 7")
    _assert_float_eq(expr(2.0, 23.0), 34)

    expr = mx.parse_f64("sin(x)^2 + 1/y")
    ref = np.sin(2) ** 2 + 1 / 4
    _assert_float_eq(expr(2.0, 4.0), ref)


def test_grad():
    expr = mx.parse_f64("x + y + z")
    with pytest.raises(TypeError):
        expr.partial(3)
    _assert_float_eq(expr.grad(1.0, 2.0, 3.0), [1, 1, 1])
    _assert_float_eq(expr.grad(np.random.random(3)), [1, 1, 1])
    _assert_float_eq(expr.grad(np.random.random(3)), [1, 1, 1])

    expr = mx.parse_f64("2*x + y^2 + cos(z) + sin(Δ)")
    ref = lambda _, y, z, Δ: (2, 2 * y, -np.sin(z), np.cos(Δ))
    x = np.random.random(4)
    _assert_float_eq(expr.grad(*x), ref(*x))


def test_hess():

    expr = mx.parse_f64("2*x * y**2 + cos(z) + sin(Δ)")
    res = expr.hess(2.0, 3.0, 4.0, 5.0)
    ref = np.array(
        [
            [0, 12, 0, 0],
            [12, 8, 0, 0],
            [0, 0, -np.cos(4), 0],
            [0, 0, 0, -np.sin(5)],
        ],
        dtype=np.float64,
    )
    _assert_float_eq(res, ref)


def test_f32():
    expr = mx.parse_f32("sin(x)^(round 2.3) + 1/y")
    ref = np.sin(2) ** 2 + 1 / 4
    tol_f32 = 1e-6
    _assert_float_eq(expr(np.asarray((2.0, 4.0), dtype=np.float32)), ref, tol=tol_f32)

    expr = mx.parse_f32("2*x * y**2 + cos(z) + sin(Δ)")
    res = expr.hess(np.asarray((2.0, 3.0, 4.0, 5.0), dtype=np.float32))
    ref = np.array(
        [
            [0, 12, 0, 0],
            [12, 8, 0, 0],
            [0, 0, -np.cos(4), 0],
            [0, 0, 0, -np.sin(5)],
        ],
        dtype=np.float32,
    )
    assert res.dtype == np.float32
    _assert_float_eq(res, ref, tol=tol_f32)


if __name__ == "__main__":
    test_eval()
    test_grad()
    test_hess()
    test_f32()
