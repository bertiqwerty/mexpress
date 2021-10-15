use std::fmt::Debug;
use std::str::FromStr;

use exmex::Express;
use exmex::OwnedFlatEx;
use num::Float;
use numpy::PyArrayDyn;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

fn partial<T: Float + Debug>(expr: &OwnedFlatEx<T>, i: i64) -> PyResult<OwnedFlatEx<T>> {
    expr.clone()
        .partial(i as usize)
        .map_err(|e| PyTypeError::new_err(e.msg))
}

fn eval<T: Float + Debug + numpy::Element>(expr: &OwnedFlatEx<T>, x: &PyArrayDyn<T>) -> PyResult<T> {
    unsafe {
        expr.eval(x.as_slice()?)
            .map_err(|e| PyTypeError::new_err(e.msg))
    }
}

fn unparse<T: Debug + Float>(expr: &OwnedFlatEx<T>) -> PyResult<String> {
    expr.unparse().map_err(|e| PyTypeError::new_err(e.msg))
}

#[pyclass]
struct InterfEx {
    expr: OwnedFlatEx<f64>,
}

#[pymethods]
impl InterfEx {
    #[call]
    fn __call__(&self, x: &PyArrayDyn<f64>) -> PyResult<f64> {
        eval(&self.expr, x)
    }

    fn partial(&self, i: i64) -> PyResult<InterfEx> {
        Ok(Self {
            expr: partial(&self.expr, i)?,
        })
    }

    fn n_vars(&self) -> PyResult<i64> {
        Ok(self.expr.n_vars() as i64)
    }

    fn unparse(&self) -> PyResult<String> {
        unparse(&self.expr)
    }
}

fn native_parse_<T>(s: &str) -> PyResult<OwnedFlatEx<T>>
where
    T: Debug + Float + FromStr,
    <T as FromStr>::Err: Debug,
{
    OwnedFlatEx::<T>::from_str(s).map_err(|e| PyTypeError::new_err(e.msg))
}

#[pyfunction]
fn native_parse(s: &str) -> PyResult<InterfEx> {
    Ok(InterfEx {
        expr: native_parse_::<f64>(s)?,
    })
}

#[pymodule]
fn mexpress(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_parse, m)?)?;
    m.add_class::<InterfEx>()?;
    Ok(())
}
