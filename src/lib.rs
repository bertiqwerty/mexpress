use std::fmt::Debug;
use std::str::FromStr;

use exmex::{Differentiate, Express, FlatEx};
use num::Float;
use numpy::PyArray1;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

pub trait DataType: Clone + FromStr + Debug + Default {}
impl<T: Clone + FromStr + Debug + Default> DataType for T {}

fn partial<T>(expr: &FlatEx<T>, i: i64) -> PyResult<FlatEx<T>>
where
    T: Float + DataType,
    <T as FromStr>::Err: Debug,
{
    expr.clone()
        .partial(i as usize)
        .map_err(|e| PyTypeError::new_err(e.msg().to_string()))
}

fn eval<T>(expr: &FlatEx<T>, x: &PyArray1<T>) -> PyResult<T>
where
    T: DataType + Float + numpy::Element,
    <T as FromStr>::Err: Debug,
{
    unsafe {
        expr.eval(x.as_slice()?)
            .map_err(|e| PyTypeError::new_err(e.msg().to_string()))
    }
}

fn unparse<T>(expr: &FlatEx<T>) -> PyResult<String> 
where
    T: Float + DataType,
    <T as FromStr>::Err: Debug,
{
    Ok(expr.unparse().to_string())
}

fn native_parse<T>(s: &str) -> PyResult<FlatEx<T>>
where
    T: DataType + Float,
    <T as FromStr>::Err: Debug,
{
    FlatEx::<T>::parse(s).map_err(|e| PyTypeError::new_err(e.msg().to_string()))
}

macro_rules! interf_ex {
    ($interf_ex_name:ident, $float:ty, $parse_name:ident) => {
        #[pyclass]
        struct $interf_ex_name {
            expr: FlatEx<$float>,
        }

        #[pymethods]
        impl $interf_ex_name {
            fn __call__(&self, x: &PyArray1<$float>) -> PyResult<$float> {
                eval(&self.expr, x)
            }

            fn partial(&self, i: i64) -> PyResult<$interf_ex_name> {
                Ok(Self {
                    expr: partial(&self.expr, i)?,
                })
            }

            fn n_vars(&self) -> PyResult<i64> {
                Ok(self.expr.var_names().len() as i64)
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
