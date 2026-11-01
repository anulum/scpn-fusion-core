// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Closed-Contour Geometry Contract
//! Closed poloidal contours and point classification in physical `(R, Z)`.
//!
//! Same contract as `scpn_fusion.core.contour_geometry`:
//! a contour is a simple closed polygon in metres; a repeated closing vertex
//! is dropped; repeated consecutive vertices, fewer than three distinct
//! vertices, zero area, spikes and self-intersections are rejected. Either
//! vertex order is accepted. A point within the tolerance of an edge is on the
//! boundary; otherwise signed even-odd crossings decide inside or outside.

use fusion_types::error::{FusionError, FusionResult};

/// Default boundary tolerance in metres (far above binary64 rounding at
/// reactor radii, far below engineering contour precision).
pub const DEFAULT_CONTOUR_TOLERANCE_M: f64 = 1.0e-9;

/// Classification of a point relative to a closed contour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLocation {
    /// Strictly inside, farther than the tolerance from every edge.
    Inside,
    /// Within the tolerance of an edge or vertex.
    OnBoundary,
    /// Strictly outside, farther than the tolerance from every edge.
    Outside,
}

/// Vertex order of a validated contour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourOrientation {
    /// Positive shoelace area.
    CounterClockwise,
    /// Negative shoelace area.
    Clockwise,
}

/// Validated simple closed contour.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedContour {
    vertices: Vec<(f64, f64)>,
    signed_area_m2: f64,
    tolerance_m: f64,
}

impl ClosedContour {
    /// Validate `points` as a closed contour with the given tolerance.
    pub fn new(points: &[(f64, f64)], tolerance_m: f64, name: &str) -> FusionResult<Self> {
        if !tolerance_m.is_finite() || tolerance_m <= 0.0 {
            return Err(config(format!(
                "{name}: tolerance_m must be finite and positive"
            )));
        }
        if points.iter().any(|(r, z)| !r.is_finite() || !z.is_finite()) {
            return Err(config(format!("{name} must contain finite values only")));
        }
        let mut vertices = points.to_vec();
        if vertices.len() >= 2 {
            let first = vertices[0];
            let last = vertices[vertices.len() - 1];
            if distance(first, last) <= tolerance_m {
                vertices.pop();
            }
        }
        if vertices.len() < 3 {
            return Err(config(format!(
                "{name} must contain at least three distinct vertices"
            )));
        }
        let n = vertices.len();
        let mut perimeter = 0.0;
        let mut twice_area = 0.0;
        for i in 0..n {
            let a = vertices[i];
            let b = vertices[(i + 1) % n];
            let step = distance(a, b);
            if step <= tolerance_m {
                return Err(config(format!("{name} repeats vertex {i} consecutively")));
            }
            perimeter += step;
            twice_area += a.0 * b.1 - b.0 * a.1;
        }
        let signed_area_m2 = 0.5 * twice_area;
        if signed_area_m2.abs() <= tolerance_m * perimeter {
            return Err(config(format!("{name} encloses zero area")));
        }
        reject_self_intersections(&vertices, tolerance_m, name)?;
        Ok(Self {
            vertices,
            signed_area_m2,
            tolerance_m,
        })
    }

    /// Distinct vertices without a repeated closing point.
    pub fn vertices(&self) -> &[(f64, f64)] {
        &self.vertices
    }

    /// Shoelace area in square metres; positive for counter-clockwise order.
    pub fn signed_area_m2(&self) -> f64 {
        self.signed_area_m2
    }

    /// Tolerance used for validation and classification.
    pub fn tolerance_m(&self) -> f64 {
        self.tolerance_m
    }

    /// Vertex order of the contour.
    pub fn orientation(&self) -> ContourOrientation {
        if self.signed_area_m2 > 0.0 {
            ContourOrientation::CounterClockwise
        } else {
            ContourOrientation::Clockwise
        }
    }

    /// Classify one point relative to the contour.
    pub fn classify(&self, point: (f64, f64)) -> PointLocation {
        let n = self.vertices.len();
        let (r, z) = point;
        let mut inside = false;
        for i in 0..n {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % n];
            if point_segment_distance(point, a, b) <= self.tolerance_m {
                return PointLocation::OnBoundary;
            }
            if (a.1 > z) != (b.1 > z) {
                let r_cross = a.0 + (z - a.1) * (b.0 - a.0) / (b.1 - a.1);
                if r < r_cross {
                    inside = !inside;
                }
            }
        }
        if inside {
            PointLocation::Inside
        } else {
            PointLocation::Outside
        }
    }
}

fn config(message: String) -> FusionError {
    FusionError::ConfigError(message)
}

fn distance(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

fn point_segment_distance(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (er, ez) = (b.0 - a.0, b.1 - a.1);
    let length_sq = er * er + ez * ez;
    let t = (((p.0 - a.0) * er + (p.1 - a.1) * ez) / length_sq).clamp(0.0, 1.0);
    distance(p, (a.0 + t * er, a.1 + t * ez))
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn reject_self_intersections(
    vertices: &[(f64, f64)],
    tolerance_m: f64,
    name: &str,
) -> FusionResult<()> {
    let n = vertices.len();
    let edge = |k: usize| (vertices[k], vertices[(k + 1) % n]);
    for i in 0..n {
        let (a0, a1) = edge(i);
        let next = (i + 1) % n;
        let (_, c1) = edge(next);
        // A spike folds the far end of one edge back onto its neighbour.
        if point_segment_distance(c1, a0, a1) <= tolerance_m
            || point_segment_distance(a0, a1, c1) <= tolerance_m
        {
            return Err(config(format!(
                "{name} folds back on itself at vertex {next}"
            )));
        }
        for j in (i + 2)..n {
            if i == 0 && j == n - 1 {
                continue;
            }
            let (b0, b1) = edge(j);
            let d1 = cross(a0, a1, b0);
            let d2 = cross(a0, a1, b1);
            let d3 = cross(b0, b1, a0);
            let d4 = cross(b0, b1, a1);
            let proper = d1.signum() * d2.signum() < 0.0 && d3.signum() * d4.signum() < 0.0;
            // Two segments that do not cross are closest at an endpoint.
            let near = point_segment_distance(b0, a0, a1)
                .min(point_segment_distance(b1, a0, a1))
                .min(point_segment_distance(a0, b0, b1))
                .min(point_segment_distance(a1, b0, b1));
            if proper || near <= tolerance_m {
                return Err(config(format!(
                    "{name} is self-intersecting: edges {i} and {j} meet"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIANGLE: [(f64, f64); 3] = [(4.0, -1.0), (6.0, -1.0), (5.0, 1.0)];

    fn contour(points: &[(f64, f64)]) -> ClosedContour {
        ClosedContour::new(points, DEFAULT_CONTOUR_TOLERANCE_M, "contour").expect("valid contour")
    }

    #[test]
    fn audit_triangle_contains_interior_point_in_both_orders() {
        let mut reversed = TRIANGLE.to_vec();
        reversed.reverse();
        for (points, orientation) in [
            (TRIANGLE.to_vec(), ContourOrientation::CounterClockwise),
            (reversed, ContourOrientation::Clockwise),
        ] {
            let c = contour(&points);
            assert_eq!(c.orientation(), orientation);
            assert_eq!(c.classify((5.0, 0.0)), PointLocation::Inside);
            assert_eq!(c.classify((4.6, 0.0)), PointLocation::Inside);
            assert_eq!(c.classify((4.4, 0.0)), PointLocation::Outside);
            assert_eq!(c.classify((5.6, 0.0)), PointLocation::Outside);
        }
    }

    #[test]
    fn edges_and_vertices_are_on_boundary_within_tolerance() {
        let c = contour(&TRIANGLE);
        let tol = DEFAULT_CONTOUR_TOLERANCE_M;
        assert_eq!(c.classify((4.0, -1.0)), PointLocation::OnBoundary);
        assert_eq!(c.classify((5.0, 1.0)), PointLocation::OnBoundary);
        assert_eq!(c.classify((5.0, -1.0)), PointLocation::OnBoundary);
        assert_eq!(c.classify((4.5, 0.0)), PointLocation::OnBoundary);
        assert_eq!(
            c.classify((5.0, -1.0 - 0.5 * tol)),
            PointLocation::OnBoundary
        );
        assert_eq!(c.classify((5.0, -1.0 - 10.0 * tol)), PointLocation::Outside);
        assert_eq!(c.classify((5.0, -1.0 + 10.0 * tol)), PointLocation::Inside);
    }

    #[test]
    fn concave_contour_excludes_its_notch_under_translation_and_scaling() {
        let base = [
            (0.0, -2.0),
            (3.0, -2.0),
            (3.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (3.0, 1.0),
            (3.0, 2.0),
            (0.0, 2.0),
        ];
        for (shift, scale) in [
            ((0.0, 0.0), 1.0),
            ((7.25, -3.5), 1.0),
            ((2.0, 1.0), 40.0),
            ((0.0, 0.0), 0.01),
        ] {
            let map = |p: (f64, f64)| (p.0 * scale + shift.0, p.1 * scale + shift.1);
            let points: Vec<(f64, f64)> = base.iter().map(|p| map(*p)).collect();
            let c = contour(&points);
            assert_eq!(c.classify(map((2.0, 0.0))), PointLocation::Outside);
            assert_eq!(c.classify(map((0.5, 0.0))), PointLocation::Inside);
            assert_eq!(c.classify(map((0.5, 1.5))), PointLocation::Inside);
            assert_eq!(c.classify(map((3.5, 0.0))), PointLocation::Outside);
            assert_eq!(c.classify(map((1.0, 0.0))), PointLocation::OnBoundary);
        }
    }

    #[test]
    fn repeated_closing_vertex_is_dropped() {
        let closed = [TRIANGLE[0], TRIANGLE[1], TRIANGLE[2], TRIANGLE[0]];
        assert_eq!(contour(&closed).vertices().len(), 3);
    }

    #[test]
    fn malformed_contours_are_rejected() {
        let cases: [(&[(f64, f64)], &str); 7] = [
            (
                &[(0.0, 0.0), (1.0, 0.0)],
                "at least three distinct vertices",
            ),
            (
                &[(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
                "repeats vertex",
            ),
            (&[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], "zero area"),
            (
                &[(0.0, 0.0), (2.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
                "folds back",
            ),
            (
                &[(0.0, 0.0), (3.0, 0.0), (0.0, 1.0), (2.0, 1.5)],
                "self-intersecting",
            ),
            (
                &[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (1.0, 0.0), (0.0, 2.0)],
                "self-intersecting",
            ),
            (&[(0.0, 0.0), (1.0, f64::NAN), (0.0, 1.0)], "finite values"),
        ];
        for (points, message) in cases {
            let err = ClosedContour::new(points, DEFAULT_CONTOUR_TOLERANCE_M, "contour")
                .expect_err("malformed contour must be rejected");
            assert!(err.to_string().contains(message), "{err} lacks {message}");
        }
        for tolerance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(ClosedContour::new(&TRIANGLE, tolerance, "contour").is_err());
        }
    }
}
