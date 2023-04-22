# Mexpress
Math parser and evaluator in Python capable of computing gradients and Hessians. Mexpress uses the Rust crate [Exmex](https://crates.io/crates/exmex).
## Installation 
[![PyPI version](https://img.shields.io/pypi/v/mexpress.svg?maxAge=3600)](https://pypi.org/project/mexpress/)
[![example workflow](https://github.com/bertiqwerty/mexpress/actions/workflows/test.yml/badge.svg)](https://github.com/bertiqwerty/mexpress)
[![dependency status](https://deps.rs/repo/github/bertiqwerty/mexpress/status.svg)](https://deps.rs/repo/github/bertiqwerty/mexpress)

```
pip install mexpress
```
## Usage

```python
import mexpress
import numpy as np

# parse function with parse_f64 or parse_f32
f = mexpress.parse_f64("(x - 1) ** 2 - y * x + 3 * y ** 2")

# evaluate function at (2, 4)
y = f(2.0, 4.0)
assert abs(y - 41) < 1e-12

# evaluate gradient at (2, 4)
grad_2_4 = f.grad(2.0, 4.0)
assert np.linalg.norm(grad_2_4 - [-2, 22]) < 1e-12

# evaluate Hessian at (2, 4)
hess_2_4 = f.hess(2.0, 4.0)
assert np.linalg.norm(hess_2_4 - [[2, -1], [-1, 6]]) < 1e-12
```

Besides `**` you can also use `^` for exponentiation. Currently, a list of supported mathematical operators can be found in the documentation of [Exmex](https://docs.rs/exmex/0.12.0/exmex/struct.FloatOpsFactory.html).

## Optimization Example

With gradients and Hessians one can at least locally optimize differentiable functions passed as strings, e.g., with `scipy.optimize`.
```Python
from scipy.optimize import minimize
import mexpress

func_str = f"(1 - x) ** 2 + 100 * (y - x ** 2) ** 2 + (z - 7) ** 2 + (ρ + 5) ** 2"
f = mexpress.parse_f64(func_str)
res = minimize(f, x0=[0.0, 0.0, 0.0, 0.0], method="trust-ncg", jac=f.grad, hess=f.hess)
```
We have played around with different methods to optimize `func_str`, see the following output of `py/demo/opt.py`. In the following table, `#fails` is the number of fails out of 100 attempts with random `x0`s. The iterations and elapsed seconds are medians.
```
CG             #fails   0   #it  44   0.0049996 sec   jac True    hess False
CG             #fails  23   #it  44   0.0179558 sec   jac False   hess False
Newton-CG      #fails   0   #it  38   0.0049839 sec   jac True    hess True
Newton-CG      #fails   5   #it  37   0.0059988 sec   jac True    hess False
trust-krylov   #fails   0   #it  31   0.0255845 sec   jac True    hess True
trust-ncg      #fails   0   #it  32   0.0030000 sec   jac True    hess True
trust-exact    #fails   0   #it  30   0.0060000 sec   jac True    hess True
BFGS           #fails   0   #it  72   0.0059998 sec   jac True    hess False
BFGS           #fails  21   #it  74   0.0169995 sec   jac False   hess False
L-BFGS-B       #fails   0   #it  43   0.0019979 sec   jac True    hess False
L-BFGS-B       #fails   0   #it  42   0.0069985 sec   jac False   hess False
Nelder-Mead    #fails   0   #it 441   0.0131288 sec   jac False   hess False
SLSQP          #fails   0   #it  34   0.0029492 sec   jac True    hess False
dogleg         #fails  17   #it  27   0.0027690 sec   jac True    hess True
TNC            #fails   0   #it  29   0.0029995 sec   jac True    hess False
TNC            #fails   0   #it  27   0.0110002 sec   jac False   hess False
COBYLA         #fails  46   #it  -1   0.0163412 sec   jac False   hess False
POWELL         #fails   0   #it  22   0.0139999 sec   jac False   hess False
```
