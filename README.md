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

## Optimization

With gradients and Hessians one can optimize functions passed as strings, e.g., with `scipy.optimize`.
```Python
from scipy.optimize import minimize
import mexpress

func_str = f"(1-x) ** 2 + 100*(y-x ** 2) ** 2 "
f = mexpress.parse(func_str)
res = minimize(f, [0.0, 0.0], method="trust-ncg", jac=f.grad, hess=f.hess)
```
We have played around with different optimizers on `func_str`, see the following output of `py/demo.py`. 
```
CG           1.0000, 1.0000 0.010 secs jac_True hess_Fals
CG           1.0000, 1.0000 0.025 secs jac_Fals hess_Fals
Newton-CG    1.0000, 0.9999 0.019 secs jac_True hess_True
Newton-CG    1.0000, 1.0000 0.024 secs jac_True hess_Fals
trust-krylov 1.0000, 1.0000 0.043 secs jac_True hess_True
trust-ncg    0.9999, 0.9998 0.007 secs jac_True hess_True
trust-exact  1.0000, 1.0000 0.011 secs jac_True hess_True
```
