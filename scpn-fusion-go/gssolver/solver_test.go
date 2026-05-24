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
