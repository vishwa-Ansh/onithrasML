
use numpy::{
    IntoPyArray,
    PyArray1,
    PyReadonlyArray1,
    PyReadonlyArray2,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use onithrasml_core::LinearRegression as RustLinearRegression;

#[pyclass(name = "LinearRegression")]
struct LinearRegression {
    model: RustLinearRegression,
}

#[pymethods]
impl LinearRegression {
    #[new]
    fn new() -> Self {
        Self {
            model: RustLinearRegression::new(),
        }
    }

    fn fit(
        &mut self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let x = x.as_array().to_owned();
        let y = y.as_array().to_owned();

        self.model
            .fit(&x, &y)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(())
    }

    fn predict(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let x = x.as_array().to_owned();

        let predictions = self.model
            .predict(&x)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(predictions.into_pyarray(py).unbind())
    }

    #[getter]
    fn coef_(
        &self,
        py: Python<'_>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let coefficients = self.model.coefficients.as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err("Model has not been fitted")
            })?;

        Ok(coefficients.clone().into_pyarray(py).unbind())
    }

    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        if self.model.coefficients.is_none() {
            return Err(PyRuntimeError::new_err(
                "Model has not been fitted"
            ));
        }

        Ok(self.model.intercept)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LinearRegression>()?;
    Ok(())
}
