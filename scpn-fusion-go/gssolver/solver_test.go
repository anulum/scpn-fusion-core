// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native Go Solver Tests
package gssolver

import (
	"math"
	"testing"
)

func referenceCase() Case {
	return Case{RMin: 1, RMax: 3, ZMin: -1.2, ZMax: 1.2, NR: 17, NZ: 17, IpTarget: 1e6, Mu0: 4e-7 * math.Pi, NPicard: 8, NJacobi: 16, Alpha: 0.1, OmegaJ: 2.0 / 3.0, BetaMix: 0.5}
}

func TestSolvePreservesPhysicsInvariants(t *testing.T) {
	result, err := Solve(referenceCase())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Psi) != 17 || len(result.Psi[0]) != 17 {
		t.Fatalf("unexpected shape")
	}
	if len(result.ResidualHistory) != 8 {
		t.Fatalf("unexpected residual history length")
	}
	maxInterior := 0.0
	for iz := range result.Psi {
		for ir, value := range result.Psi[iz] {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				t.Fatalf("non-finite psi")
			}
			if iz == 0 || iz == 16 || ir == 0 || ir == 16 {
				if math.Abs(value) > 1e-14 {
					t.Fatalf("boundary changed")
				}
			} else {
				maxInterior = math.Max(maxInterior, math.Abs(value))
			}
		}
	}
	if maxInterior <= 1e-6 {
		t.Fatalf("interior solve is trivial")
	}
}

func TestCaseValidation(t *testing.T) {
	cases := []Case{referenceCase(), referenceCase(), referenceCase(), referenceCase()}
	cases[0].NR = 2
	cases[1].Alpha = 0
	cases[2].OmegaJ = 2
	cases[3].BetaMix = 1.5
	for _, c := range cases {
		if err := c.Validate(); err == nil {
			t.Fatalf("expected invalid case")
		}
	}
}

func TestOperatorCurrentClosureManufacturedZQuadratic(t *testing.T) {
	c := referenceCase()
	c.ZMin = -1.0
	c.ZMax = 1.0
	c.NR = 17
	c.NZ = 19
	_, z, _, dR, dZ := grid(c)
	coeff := -0.25
	psi := zeros(c.NZ, c.NR)
	for iz := 0; iz < c.NZ; iz++ {
		for ir := 0; ir < c.NR; ir++ {
			psi[iz][ir] = coeff * z[iz] * z[iz]
		}
	}

	deltaStar, err := DeltaStar(c, psi)
	if err != nil {
		t.Fatal(err)
	}
	currentDensity, err := ToroidalCurrentDensityFromFlux(c, psi)
	if err != nil {
		t.Fatal(err)
	}
	totalCurrent, err := TotalToroidalCurrentFromFlux(c, psi)
	if err != nil {
		t.Fatal(err)
	}

	expectedTotal := 0.0
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			r := c.RMin + float64(ir)*dR
			expectedJ := -2.0 * coeff / (c.Mu0 * r)
			if math.Abs(deltaStar[iz][ir]-2.0*coeff) > 1.0e-12 {
				t.Fatalf("unexpected Delta* at (%d, %d): %.17g", iz, ir, deltaStar[iz][ir])
			}
			if math.Abs(currentDensity[iz][ir]-expectedJ) > 1.0e-6 {
				t.Fatalf("unexpected J_phi at (%d, %d): %.17g", iz, ir, currentDensity[iz][ir])
			}
			expectedTotal += expectedJ * dR * dZ
		}
	}
	if math.Abs((totalCurrent-expectedTotal)/expectedTotal) > 1.0e-12 {
		t.Fatalf("unexpected total current: got %.17g expected %.17g", totalCurrent, expectedTotal)
	}

	mask := make([][]bool, c.NZ)
	for iz := 0; iz < c.NZ; iz++ {
		mask[iz] = make([]bool, c.NR)
		for ir := 0; ir < c.NR; ir++ {
			mask[iz][ir] = iz > 2 && iz < c.NZ-3 && ir > 3 && ir < c.NR-4
		}
	}
	maskedCurrent, err := TotalToroidalCurrentFromFluxMasked(c, psi, mask)
	if err != nil {
		t.Fatal(err)
	}
	expectedMaskedTotal := 0.0
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			if mask[iz][ir] {
				r := c.RMin + float64(ir)*dR
				expectedMaskedTotal += -2.0 * coeff / (c.Mu0 * r) * dR * dZ
			}
		}
	}
	if math.Abs((maskedCurrent-expectedMaskedTotal)/expectedMaskedTotal) > 1.0e-12 {
		t.Fatalf("unexpected masked total current: got %.17g expected %.17g", maskedCurrent, expectedMaskedTotal)
	}
	if math.Abs(maskedCurrent) >= math.Abs(totalCurrent) {
		t.Fatalf("masked current should be smaller than full-domain current")
	}
	if _, err := TotalToroidalCurrentFromFluxMasked(c, psi, nil); err == nil {
		t.Fatalf("empty current mask must fail")
	}

	radialCoeff := 0.03125
	verticalCoeff := -0.125
	psiRadial := zeros(c.NZ, c.NR)
	for iz := 0; iz < c.NZ; iz++ {
		for ir := 0; ir < c.NR; ir++ {
			r := c.RMin + float64(ir)*dR
			psiRadial[iz][ir] = radialCoeff*r*r*r*r + verticalCoeff*z[iz]*z[iz]
		}
	}
	deltaStarRadial, err := DeltaStar(c, psiRadial)
	if err != nil {
		t.Fatal(err)
	}
	currentDensityRadial, err := ToroidalCurrentDensityFromFlux(c, psiRadial)
	if err != nil {
		t.Fatal(err)
	}
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			r := c.RMin + float64(ir)*dR
			expectedDelta := 8.0*radialCoeff*r*r + 2.0*verticalCoeff - 2.0*radialCoeff*dR*dR
			expectedJ := -expectedDelta / (c.Mu0 * r)
			if math.Abs(deltaStarRadial[iz][ir]-expectedDelta) > 1.0e-12 {
				t.Fatalf("unexpected radial Delta* at (%d, %d): %.17g", iz, ir, deltaStarRadial[iz][ir])
			}
			if math.Abs(currentDensityRadial[iz][ir]-expectedJ) > 1.0e-6 {
				t.Fatalf("unexpected radial J_phi at (%d, %d): %.17g", iz, ir, currentDensityRadial[iz][ir])
			}
		}
	}

	mixedCoeff := 0.05
	psiMixed := zeros(c.NZ, c.NR)
	for iz := 0; iz < c.NZ; iz++ {
		for ir := 0; ir < c.NR; ir++ {
			r := c.RMin + float64(ir)*dR
			psiMixed[iz][ir] = mixedCoeff*r*r*z[iz]*z[iz] + verticalCoeff*z[iz]*z[iz]
		}
	}
	deltaStarMixed, err := DeltaStar(c, psiMixed)
	if err != nil {
		t.Fatal(err)
	}
	currentDensityMixed, err := ToroidalCurrentDensityFromFlux(c, psiMixed)
	if err != nil {
		t.Fatal(err)
	}
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			r := c.RMin + float64(ir)*dR
			expectedDelta := 2.0*mixedCoeff*r*r + 2.0*verticalCoeff
			expectedJ := -expectedDelta / (c.Mu0 * r)
			if math.Abs(deltaStarMixed[iz][ir]-expectedDelta) > 1.0e-12 {
				t.Fatalf("unexpected mixed Delta* at (%d, %d): %.17g", iz, ir, deltaStarMixed[iz][ir])
			}
			if math.Abs(currentDensityMixed[iz][ir]-expectedJ) > 1.0e-6 {
				t.Fatalf("unexpected mixed J_phi at (%d, %d): %.17g", iz, ir, currentDensityMixed[iz][ir])
			}
		}
	}
}
