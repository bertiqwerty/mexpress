use numpy::PyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::exceptions::PyTypeError;
use exmex::OwnedFlatEx;
use exmex::Express;
#[pyclass]
struct FlatEx {
    owned_flatex: OwnedFlatEx::<f64>
}

#[pymethods]
impl FlatEx {

    #[call]
    fn __call__(&self, x: &PyArrayDyn<f64>) -> PyResult<f64> {
        unsafe {
            Ok(self.owned_flatex.eval(x.as_slice()?).map_err(|e|PyTypeError::new_err(e.msg))?)
        }
    }
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn parse(s: &str) -> PyResult<FlatEx> {
    Ok(FlatEx{
        owned_flatex: OwnedFlatEx::<f64>::from_str(s).map_err(|e|PyTypeError::new_err(e.msg))?
    })
}


#[pymodule]
fn mexpress(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse, m)?)?;
    m.add_class::<FlatEx>()?;
    Ok(())
}