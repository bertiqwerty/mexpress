use std::fmt::Debug;
use std::str::FromStr;

use exmex::Express;
use exmex::OwnedFlatEx;
use num::Float;
use numpy::PyArray1;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

fn partial<T: Float + Debug>(expr: &OwnedFlatEx<T>, i: i64) -> PyResult<OwnedFlatEx<T>> {
    expr.clone()
        .partial(i as usize)
        .map_err(|e| PyTypeError::new_err(e.msg))
}

fn eval<T: Float + Debug + numpy::Element>(expr: &OwnedFlatEx<T>, x: &PyArray1<T>) -> PyResult<T> {
    unsafe {
        expr.eval(x.as_slice()?)
            .map_err(|e| PyTypeError::new_err(e.msg))
    }
}

fn unparse<T: Debug + Float>(expr: &OwnedFlatEx<T>) -> PyResult<String> {
    expr.unparse().map_err(|e| PyTypeError::new_err(e.msg))
}

fn native_parse<T>(s: &str) -> PyResult<OwnedFlatEx<T>>
where
    T: Debug + Float + FromStr,
    <T as FromStr>::Err: Debug,
{
    OwnedFlatEx::<T>::from_str(s).map_err(|e| PyTypeError::new_err(e.msg))
}

macro_rules! interf_ex {
    ($interf_ex_name:ident, $float:ty, $parse_name:ident) => {
        #[pyclass]
        struct $interf_ex_name {
            expr: OwnedFlatEx<$float>,
        }

        #[pymethods]
        impl $interf_ex_name {
            #[call]
            fn __call__(&self, x: &PyArray1<$float>) -> PyResult<$float> {
                eval(&self.expr, x)
            }

            fn partial(&self, i: i64) -> PyResult<$interf_ex_name> {
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
        #[pyfunction]
        fn $parse_name(s: &str) -> PyResult<$interf_ex_name> {
            Ok($interf_ex_name {
                expr: native_parse::<$float>(s)?,
            })
        }
    };
}

interf_ex!(InterfExF64, f64, native_parse_f64);
interf_ex!(InterfExF32, f32, native_parse_f32);

#[pymodule]
fn mexpress(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_parse_f64, m)?)?;
    m.add_class::<InterfExF64>()?;
    m.add_function(wrap_pyfunction!(native_parse_f32, m)?)?;
    m.add_class::<InterfExF32>()?;
    Ok(())
}
