# Mexpress
Math parser and evaluator based on the Rust crate Exmex.
## Installation 

```
pip install mexpress
```
## Usage

```python
import mexpress
import numpy as np

# parse function
f = mexpress.parse("(x - 1) ** 2 - y * x + 3 * y ** 2")

# evaluate function at (2, 4)
y = f(2, 4)
assert abs(y + 3) < 1e-12

# evaluate gradient at (2, 4)
grad_2_4 = f.grad(2, 4)
assert np.linalg.norm(grad_2_4 - [-2, 22]) < 1e-12

# evaluate Hessian at (2, 4)
hess_2_4 = f.hess(2, 4)
assert np.linalg.norm(hess_2_4 - [[2, -1], [-1, 6]]) < 1e-12
```

## Optimization Example

With gradients and Hessians one can at least locally optimize differentiable functions passed as strings, e.g., with `scipy.optimize`.
```Python
from scipy.optimize import minimize
import mexpress

func_str = "(1-x) ** 2 + 100*(y-x ** 2) ** 2"
f = mexpress.parse(func_str)
res = minimize(f, [0.0, 0.0], method="trust-ncg", jac=f.grad, hess=f.hess)
```
We have played around with different optimizers on `func_str`, see the following output of [`py/demo.py`](https://github.com/bertiqwerty/mexpress/blob/main/py/demo.py). 
```
CG             (0.99999, 0.99998)   0.010 secs   jac True    hess False
CG             (1.00000, 0.99999)   0.019 secs   jac False   hess False
Newton-CG      (0.99996, 0.99992)   0.022 secs   jac True    hess True 
Newton-CG      (1.00000, 1.00000)   0.015 secs   jac True    hess False
trust-krylov   (1.00000, 1.00000)   0.032 secs   jac True    hess True 
trust-ncg      (0.99991, 0.99983)   0.007 secs   jac True    hess True
trust-exact    (1.00000, 1.00000)   0.009 secs   jac True    hess True
```
