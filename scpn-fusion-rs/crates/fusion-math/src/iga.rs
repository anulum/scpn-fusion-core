//! Reduced isogeometric-analysis helpers (NURBS curve evaluation and sampling).

/// 2D control point used for boundary curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlPoint2D {
    pub x: f64,
    pub y: f64,
}

/// Open NURBS curve in 2D.
#[derive(Debug, Clone)]
pub struct NurbsCurve2D {
    degree: usize,
    knots: Vec<f64>,
    control_points: Vec<ControlPoint2D>,
    weights: Vec<f64>,
}

impl NurbsCurve2D {
    /// Validate and create a NURBS curve.
    pub fn new(
        degree: usize,
        knots: Vec<f64>,
        control_points: Vec<ControlPoint2D>,
        weights: Vec<f64>,
    ) -> Result<Self, String> {
        if control_points.is_empty() {
            return Err("NURBS requires at least one control point".to_string());
        }
        if control_points.len() != weights.len() {
            return Err("Control points and weights length mismatch".to_string());
        }
        let expected_knots = control_points.len() + degree + 1;
        if knots.len() != expected_knots {
            return Err(format!(
                "Invalid knot vector length: expected {expected_knots}, got {}",
                knots.len()
            ));
        }
        if knots.windows(2).any(|w| w[1] < w[0]) {
            return Err("Knot vector must be non-decreasing".to_string());
        }
        if weights.iter().any(|w| *w <= 0.0 || !w.is_finite()) {
            return Err("NURBS weights must be positive finite values".to_string());
        }
        Ok(Self {
            degree,
            knots,
            control_points,
            weights,
        })
    }

    /// Evaluate curve point at parameter `u`.
    pub fn evaluate(&self, u: f64) -> ControlPoint2D {
        let n = self.control_points.len();
        if n == 1 {
            return self.control_points[0];
        }

        let u_min = self.knots[self.degree];
        let u_max = self.knots[self.knots.len() - self.degree - 1];
        let u_eval = u.clamp(u_min, u_max);
        if (u_eval - u_min).abs() < 1e-14 {
            return self.control_points[0];
        }
        if (u_eval - u_max).abs() < 1e-14 {
            return *self
                .control_points
                .last()
                .unwrap_or(&self.control_points[0]);
        }

        let mut numerator_x = 0.0;
        let mut numerator_y = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            let b = basis_function(i, self.degree, u_eval, &self.knots);
            let w = self.weights[i];
            let r = b * w;
            numerator_x += r * self.control_points[i].x;
            numerator_y += r * self.control_points[i].y;
            denominator += r;
        }

        if denominator.abs() < 1e-14 {
            return self.control_points[0];
        }
        ControlPoint2D {
            x: numerator_x / denominator,
            y: numerator_y / denominator,
        }
    }

    /// Sample the curve with approximately uniform parameter spacing.
    pub fn sample_uniform(&self, n_samples: usize) -> Vec<ControlPoint2D> {
        if n_samples == 0 {
            return Vec::new();
        }
        if n_samples == 1 {
            return vec![self.evaluate(self.knots[self.degree])];
        }

        let u_min = self.knots[self.degree];
        let u_max = self.knots[self.knots.len() - self.degree - 1];
        let mut out = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / (n_samples - 1) as f64;
            let u = u_min + t * (u_max - u_min);
            out.push(self.evaluate(u));
        }
        out
    }
}

/// Generate an open-uniform knot vector.
pub fn open_uniform_knots(n_control_points: usize, degree: usize) -> Vec<f64> {
    if n_control_points == 0 {
        return Vec::new();
    }
    let m = n_control_points + degree + 1;
    let mut knots = vec![0.0; m];
    for value in knots.iter_mut().take(m).skip(m - degree - 1) {
        *value = 1.0;
    }
    if m > 2 * (degree + 1) {
        let interior = m - 2 * (degree + 1);
        for j in 0..interior {
            knots[degree + 1 + j] = (j + 1) as f64 / (interior + 1) as f64;
        }
    }
    knots
}

fn basis_function(i: usize, p: usize, u: f64, knots: &[f64]) -> f64 {
    if p == 0 {
        let left = knots[i];
        let right = knots[i + 1];
        let is_last_span = (u - knots[knots.len() - 1]).abs() < 1e-14
            && (right - knots[knots.len() - 1]).abs() < 1e-14;
        if (left <= u && u < right) || is_last_span {
            1.0
        } else {
            0.0
        }
    } else {
        let left_denom = knots[i + p] - knots[i];
        let right_denom = knots[i + p + 1] - knots[i + 1];

        let left = if left_denom.abs() > 0.0 {
            (u - knots[i]) / left_denom * basis_function(i, p - 1, u, knots)
        } else {
            0.0
        };
        let right = if right_denom.abs() > 0.0 {
            (knots[i + p + 1] - u) / right_denom * basis_function(i + 1, p - 1, u, knots)
        } else {
            0.0
        };
        left + right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_uniform_knots_length() {
        let knots = open_uniform_knots(10, 3);
        assert_eq!(knots.len(), 14);
        assert!(knots.windows(2).all(|w| w[1] >= w[0]));
    }

    #[test]
    fn test_nurbs_evaluate_endpoints_for_line() {
        let degree = 3;
        let cps = vec![
            ControlPoint2D { x: 0.0, y: 0.0 },
            ControlPoint2D { x: 1.0, y: 0.0 },
            ControlPoint2D { x: 2.0, y: 0.0 },
            ControlPoint2D { x: 3.0, y: 0.0 },
        ];
        let weights = vec![1.0; cps.len()];
        let knots = open_uniform_knots(cps.len(), degree);
        let curve = NurbsCurve2D::new(degree, knots, cps, weights).unwrap();
        let start = curve.evaluate(0.0);
        let end = curve.evaluate(1.0);
        assert!((start.x - 0.0).abs() < 1e-12);
        assert!((end.x - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_nurbs_uniform_sampling_has_requested_size() {
        let degree = 2;
        let cps = vec![
            ControlPoint2D { x: 0.0, y: 0.0 },
            ControlPoint2D { x: 1.0, y: 1.0 },
            ControlPoint2D { x: 2.0, y: 0.0 },
            ControlPoint2D { x: 3.0, y: 1.0 },
            ControlPoint2D { x: 4.0, y: 0.0 },
        ];
        let weights = vec![1.0; cps.len()];
        let knots = open_uniform_knots(cps.len(), degree);
        let curve = NurbsCurve2D::new(degree, knots, cps, weights).unwrap();
        let sampled = curve.sample_uniform(64);
        assert_eq!(sampled.len(), 64);
        assert!(sampled.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }
}
