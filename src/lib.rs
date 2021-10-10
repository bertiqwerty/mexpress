use exmex::Express;
use exmex::OwnedFlatEx;
use numpy::PyArrayDyn;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
#[pyclass]
struct FlatEx {
    owned_flatex: OwnedFlatEx<f64>,
}

#[pymethods]
impl FlatEx {
    #[call]
    fn __call__(&self, x: &PyArrayDyn<f64>) -> PyResult<f64> {
        unsafe {
            Ok(self
                .owned_flatex
                .eval(x.as_slice()?)
                .map_err(|e| PyTypeError::new_err(e.msg))?)
        }
    }

    fn partial(&self, i: i64) -> PyResult<FlatEx> {
        let partial = self
            .owned_flatex
            .clone()
            .partial(i as usize)
            .map_err(|e| PyTypeError::new_err(e.msg))?;
        Ok(FlatEx {
            owned_flatex: partial,
        })
    }

    fn n_vars(&self) -> PyResult<i64> {
        Ok(self.owned_flatex.n_vars() as i64)
    }

    fn unparse(&self) -> PyResult<String> {
        Ok(self
            .owned_flatex
            .unparse()
            .map_err(|e| PyTypeError::new_err(e.msg))?)
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn native_parse(s: &str) -> PyResult<FlatEx> {
    Ok(FlatEx {
        owned_flatex: OwnedFlatEx::<f64>::from_str(s).map_err(|e| PyTypeError::new_err(e.msg))?,
    })
}

#[pymodule]
fn mexpress(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_parse, m)?)?;
    m.add_class::<FlatEx>()?;
    Ok(())
}
