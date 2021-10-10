# Mexpress
Math parser and evaluator based on the Rust crate Exmex.
## Installation 

```
pip install mexpress
```
## Basic Usage

```python
import numpy as np
import mexpress

# parse function
f = mexpress.parse("(x - 1) ** 2 - y*x + 3 * y ** 2")

# evaluate function at (2, 4)
y = f(2, 4)
assert abs(y + 3) < 1e-12

# evaluate gradient at (2, 4)
y_ = f.grad(2, 4)
assert np.linalg.norm(y_ - [-2, 22])

# evaluate Hessian at (2, 4)
y__ = f.hess(2, 4)
assert np.linalg.norm(y__ - [[2, -1],[-1, 6]])
```
