
use ndarray::{Array1, Array2, Axis};

use crate::error::{OnithrasError, Result};

pub struct LinearRegression {
    pub coefficients: Option<Array1<f64>>,
    pub intercept: f64,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
        }
    }

    pub fn fit(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<()> {
        if x.nrows() != y.len() || x.nrows() == 0 || x.ncols() == 0 {
            return Err(OnithrasError::DimensionMismatch);
        }

        let x_mean = x.mean_axis(Axis(0))
            .ok_or(OnithrasError::DimensionMismatch)?;

        let y_mean = y.mean()
            .ok_or(OnithrasError::DimensionMismatch)?;

        let centered_x = x - &x_mean;
        let centered_y = y - y_mean;

        let xtx = centered_x.t().dot(&centered_x);
        let xty = centered_x.t().dot(&centered_y);

        let coefficients = nalgebra::DMatrix::from_fn(
            xtx.nrows(),
            xtx.ncols(),
            |i, j| xtx[[i, j]],
        );

        let target = nalgebra::DVector::from_iterator(
            xty.len(),
            xty.iter().copied(),
        );

        let solution = coefficients
            .lu()
            .solve(&target)
            .ok_or_else(|| {
                OnithrasError::InvalidInput(
                    "Singular matrix: unable to solve regression".into()
                )
            })?;

        let coefficients = Array1::from_iter(solution.iter().copied());

        self.intercept = y_mean - x_mean.dot(&coefficients);
        self.coefficients = Some(coefficients);

        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let coefficients = self.coefficients.as_ref()
            .ok_or(OnithrasError::ModelNotFitted)?;

        if x.ncols() != coefficients.len() {
            return Err(OnithrasError::DimensionMismatch);
        }

        Ok(x.dot(coefficients) + self.intercept)
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}
