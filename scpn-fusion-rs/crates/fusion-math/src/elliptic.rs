// ─────────────────────────────────────────────────────────────────────
// SCPN Fusion Core — Elliptic
// © 1998–2026 Miroslav Šotek. All rights reserved.
// Contact: www.anulum.li | protoscience@anulum.li
// ORCID: https://orcid.org/0009-0009-3560-0851
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Complete elliptic integrals K(m) and E(m).
//!
//! Uses Abramowitz & Stegun polynomial approximations (Handbook of
//! Mathematical Functions, 17.3.34 and 17.3.36). Parameter convention
//! matches scipy: m = k^2 where 0 <= m < 1.

/// Complete elliptic integral of the first kind K(m).
///
/// Parameter m = k^2, where 0 <= m < 1.
/// Matches `scipy.special.ellipk(m)`.
///
/// Accuracy: |error| < 2e-8 for 0 <= m < 1.
pub fn ellipk(m: f64) -> f64 {
    debug_assert!(
        (0.0..1.0).contains(&m),
        "ellipk requires 0 <= m < 1, got {m}"
    );

    let m1 = 1.0 - m;

    // Polynomial coefficients (A&S 17.3.34)
    let a0 = 1.386_294_361_12;
    let a1 = 0.096_663_442_59;
    let a2 = 0.035_900_923_83;
    let a3 = 0.037_425_637_13;
    let a4 = 0.014_511_962_12;

    let b0 = 0.5;
    let b1 = 0.124_985_935_97;
    let b2 = 0.068_802_485_76;
    let b3 = 0.033_283_553_46;
    let b4 = 0.004_417_870_12;

    let poly_a = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)));
    let poly_b = b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)));

    poly_a + poly_b * (-m1.ln())
}

/// Complete elliptic integral of the second kind E(m).
///
/// Parameter m = k^2, where 0 <= m <= 1.
/// Matches `scipy.special.ellipe(m)`.
///
/// Accuracy: |error| < 2e-8 for 0 <= m <= 1.
pub fn ellipe(m: f64) -> f64 {
    debug_assert!(
        (0.0..=1.0).contains(&m),
        "ellipe requires 0 <= m <= 1, got {m}"
    );

    if m >= 1.0 {
        return 1.0;
    }

    let m1 = 1.0 - m;

    // Polynomial coefficients (A&S 17.3.36)
    let a1 = 0.443_251_414_63;
    let a2 = 0.062_606_012_20;
    let a3 = 0.047_573_835_46;
    let a4 = 0.017_365_064_51;

    let b1 = 0.249_983_683_10;
    let b2 = 0.092_001_800_37;
    let b3 = 0.040_696_975_26;
    let b4 = 0.005_264_496_39;

    let poly_a = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)));
    let poly_b = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)));

    poly_a + poly_b * (-m1.ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from scipy.special (via reference_elliptic.json)
    #[test]
    fn test_ellipk_at_zero() {
        let expected = std::f64::consts::FRAC_PI_2;
        assert!((ellipk(0.0) - expected).abs() < 1e-8, "K(0) = pi/2");
    }

    #[test]
    fn test_ellipk_reference_values() {
        let cases: &[(f64, f64)] = &[
            (0.0, std::f64::consts::FRAC_PI_2),
            (0.1, 1.6124413487202192),
            (0.2, 1.659623598610528),
            (0.3, 1.713889448178791),
            (0.4, 1.7775193714912534),
            (0.5, 1.8540746773013719),
            (0.6, 1.9495677498060258),
            (0.7, 2.075363135292469),
            (0.8, 2.257205326820854),
            (0.9, 2.5780921133481733),
            (0.95, 2.9083372484445515),
            (0.99, 3.6956373629898747),
            (0.999, 4.841132560550296),
        ];
        for &(m, expected) in cases {
            let got = ellipk(m);
            let err = (got - expected).abs();
            assert!(
                err < 5e-8,
                "K({m}) = {got}, expected {expected}, error = {err}"
            );
        }
    }

    #[test]
    fn test_ellipe_at_zero() {
        let expected = std::f64::consts::FRAC_PI_2;
        assert!((ellipe(0.0) - expected).abs() < 1e-8, "E(0) = pi/2");
    }

    #[test]
    fn test_ellipe_reference_values() {
        let cases: &[(f64, f64)] = &[
            (0.0, std::f64::consts::FRAC_PI_2),
            (0.1, 1.5307576368977633),
            (0.2, 1.489035058095853),
            (0.3, 1.4453630644126654),
            (0.4, 1.3993921388974322),
            (0.5, 1.3506438810476755),
            (0.6, 1.2984280350469133),
            (0.7, 1.2416705679458229),
            (0.8, 1.1784899243278386),
            (0.9, 1.1047747327040733),
            (0.95, 1.0604737277662784),
            (0.99, 1.015993545025224),
            (0.999, 1.0021707908344453),
        ];
        for &(m, expected) in cases {
            let got = ellipe(m);
            let err = (got - expected).abs();
            assert!(
                err < 5e-8,
                "E({m}) = {got}, expected {expected}, error = {err}"
            );
        }
    }

    #[test]
    fn test_ellipe_at_one() {
        assert!((ellipe(1.0) - 1.0).abs() < 1e-10, "E(1) = 1");
    }
}
