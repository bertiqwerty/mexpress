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
