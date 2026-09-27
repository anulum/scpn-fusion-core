// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native Go Grad-Shafranov Solver
package gssolver

import (
	"fmt"
	"math"
)

// Case defines the grid, physical constants, and iteration controls for one
// fixed-boundary Grad-Shafranov solve.
type Case struct {
	// RMin and RMax bound the major-radius grid in metres.
	RMin, RMax, ZMin, ZMax float64
	// NR and NZ are the radial and vertical grid-point counts.
	NR, NZ int
	// IpTarget is the target toroidal plasma current and Mu0 is vacuum permeability.
	IpTarget, Mu0 float64
	// NPicard and NJacobi control the outer and inner iteration counts.
	NPicard, NJacobi int
	// Alpha, OmegaJ, and BetaMix control relaxation and source-profile mixing.
	Alpha, OmegaJ, BetaMix float64
}

// Result contains the row-major poloidal-flux grid and outer-iteration
// residual history returned by Solve.
type Result struct {
	// Psi is indexed as Psi[z][r] on the case grid.
	Psi [][]float64
	// ResidualHistory records the maximum absolute flux update per Picard step.
	ResidualHistory []float64
}

// Solve runs the native fixed-boundary Picard/Jacobi reference algorithm.
func Solve(c Case) (Result, error) {
	if err := c.Validate(); err != nil {
		return Result{}, err
	}
	r, _, rr, dR, dZ := grid(c)
	_ = r
	if !finite(dR*dR) || !finite(dZ*dZ) || dR*dR == 0 || dZ*dZ == 0 {
		return Result{}, fmt.Errorf("solver spacing arithmetic became non-finite or degenerate")
	}
	psi, err := initialPsi(c, rr)
	if err != nil {
		return Result{}, err
	}
	if err := validateFluxMatrix(c, psi); err != nil {
		return Result{}, err
	}
	residuals := make([]float64, 0, c.NPicard)
	for outer := 0; outer < c.NPicard; outer++ {
		source, err := computeSource(c, psi, rr, dR, dZ)
		if err != nil {
			return Result{}, err
		}
		psiElliptic := clone(psi)
		for inner := 0; inner < c.NJacobi; inner++ {
			psiElliptic, err = jacobiStep(c, psiElliptic, source, rr, dR, dZ)
			if err != nil {
				return Result{}, err
			}
		}
		psiNext := zeros(c.NZ, c.NR)
		maxChange := 0.0
		for iz := 0; iz < c.NZ; iz++ {
			for ir := 0; ir < c.NR; ir++ {
				psiNext[iz][ir] = (1-c.Alpha)*psi[iz][ir] + c.Alpha*psiElliptic[iz][ir]
				change := math.Abs(psiNext[iz][ir] - psi[iz][ir])
				if !finite(psiNext[iz][ir]) || !finite(change) {
					return Result{}, fmt.Errorf("flux or update residual became non-finite")
				}
				maxChange = math.Max(maxChange, change)
			}
		}
		residuals = append(residuals, maxChange)
		psi = psiNext
	}
	applyZeroBoundary(psi)
	return Result{Psi: psi, ResidualHistory: residuals}, nil
}

func grid(c Case) ([]float64, []float64, [][]float64, float64, float64) {
	r := linspace(c.RMin, c.RMax, c.NR)
	z := linspace(c.ZMin, c.ZMax, c.NZ)
	rr := zeros(c.NZ, c.NR)
	for iz := range rr {
		for ir := range rr[iz] {
			rr[iz][ir] = r[ir]
		}
	}
	return r, z, rr, r[1] - r[0], z[1] - z[0]
}

func linspace(start, stop float64, count int) []float64 {
	out := make([]float64, count)
	step := (stop - start) / float64(count-1)
	for i := range out {
		out[i] = start + step*float64(i)
	}
	return out
}

func zeros(nz, nr int) [][]float64 {
	m := make([][]float64, nz)
	for iz := range m {
		m[iz] = make([]float64, nr)
	}
	return m
}

func clone(in [][]float64) [][]float64 {
	out := make([][]float64, len(in))
	for i := range in {
		out[i] = append([]float64(nil), in[i]...)
	}
	return out
}

// initialPsi checks Gaussian arithmetic before exponentiation can hide overflow.
func initialPsi(c Case, rr [][]float64) ([][]float64, error) {
	psi := zeros(c.NZ, c.NR)
	rCenter := 0.5 * (c.RMin + c.RMax)
	if !finite(rCenter) {
		return nil, fmt.Errorf("initial Gaussian centre became non-finite")
	}
	for iz := range psi {
		for ir := range psi[iz] {
			delta := rr[iz][ir] - rCenter
			exponent := -(delta * delta) / 0.5
			if !finite(exponent) {
				return nil, fmt.Errorf("initial Gaussian exponent became non-finite")
			}
			psi[iz][ir] = math.Exp(exponent) * 0.01
		}
	}
	applyZeroBoundary(psi)
	return psi, nil
}

func applyZeroBoundary(psi [][]float64) {
	nz := len(psi)
	nr := len(psi[0])
	for ir := 0; ir < nr; ir++ {
		psi[0][ir] = 0
		psi[nz-1][ir] = 0
	}
	for iz := 0; iz < nz; iz++ {
		psi[iz][0] = 0
		psi[iz][nr-1] = 0
	}
}

// computeSource builds the current profile, refusing non-finite intermediate arithmetic.
func computeSource(c Case, psi, rr [][]float64, dR, dZ float64) ([][]float64, error) {
	psiAxis := psi[1][1]
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			psiAxis = math.Max(psiAxis, psi[iz][ir])
		}
	}
	denom := -psiAxis
	if math.Abs(denom) < 1e-9 {
		if denom == 0 {
			denom = 1e-9
		} else {
			denom = math.Copysign(1e-9, denom)
		}
	}
	jRaw := zeros(c.NZ, c.NR)
	current := 0.0
	for iz := range psi {
		for ir := range psi[iz] {
			psiNorm := (psi[iz][ir] - psiAxis) / denom
			if !finite(psiNorm) {
				return nil, fmt.Errorf("normalised flux became non-finite")
			}
			psiNorm = math.Min(1, math.Max(0, psiNorm))
			profile := 0.0
			if psiNorm >= 0 && psiNorm < 1 {
				profile = 1 - psiNorm
			}
			rSafe := math.Max(rr[iz][ir], 1e-10)
			jP := rr[iz][ir] * profile
			denominator := c.Mu0 * rSafe
			if !finite(denominator) || denominator == 0 {
				return nil, fmt.Errorf("source denominator became non-finite or zero")
			}
			jF := profile / denominator
			jRaw[iz][ir] = c.BetaMix*jP + (1-c.BetaMix)*jF
			if !finite(jP) || !finite(jF) || !finite(jRaw[iz][ir]) {
				return nil, fmt.Errorf("current profile became non-finite")
			}
			current += jRaw[iz][ir] * dR * dZ
			if !finite(current) {
				return nil, fmt.Errorf("profile current became non-finite")
			}
		}
	}
	scale := c.IpTarget / math.Max(math.Abs(current), 1e-9)
	if !finite(scale) {
		return nil, fmt.Errorf("current scaling became non-finite")
	}
	source := zeros(c.NZ, c.NR)
	for iz := range source {
		for ir := range source[iz] {
			scaledCurrent := jRaw[iz][ir] * scale
			if !finite(scaledCurrent) {
				return nil, fmt.Errorf("scaled current became non-finite")
			}
			source[iz][ir] = -c.Mu0 * rr[iz][ir] * scaledCurrent
			if !finite(source[iz][ir]) {
				return nil, fmt.Errorf("GS source became non-finite")
			}
		}
	}
	return source, nil
}

// jacobiStep applies one checked relaxation step without hiding numerical failure.
func jacobiStep(c Case, psi, source, rr [][]float64, dR, dZ float64) ([][]float64, error) {
	out := clone(psi)
	dR2 := dR * dR
	dZ2 := dZ * dZ
	aNS := 1 / dZ2
	aC := 2/dR2 + 2/dZ2
	if !finite(aNS) || !finite(aC) || aNS <= 0 || aC <= 0 {
		return nil, fmt.Errorf("Jacobi coefficient arithmetic failed")
	}
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			rSafe := math.Max(rr[iz][ir], 1e-10)
			aE := 1/dR2 - 1/(2*rSafe*dR)
			aW := 1/dR2 + 1/(2*rSafe*dR)
			if !finite(2*rSafe*dR) || 2*rSafe*dR == 0 || !finite(aE) || !finite(aW) {
				return nil, fmt.Errorf("radial Jacobi coefficient arithmetic failed")
			}
			update := (aE*psi[iz][ir+1] + aW*psi[iz][ir-1] + aNS*(psi[iz-1][ir]+psi[iz+1][ir]) - source[iz][ir]) / aC
			out[iz][ir] = (1-c.OmegaJ)*psi[iz][ir] + c.OmegaJ*update
			if !finite(update) || !finite(out[iz][ir]) {
				return nil, fmt.Errorf("Jacobi flux became non-finite")
			}
		}
	}
	return out, nil
}

func validateFluxMatrix(c Case, psi [][]float64) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(psi) != c.NZ {
		return fmt.Errorf("psi row count mismatch")
	}
	for _, row := range psi {
		if len(row) != c.NR {
			return fmt.Errorf("psi column count mismatch")
		}
		for _, value := range row {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return fmt.Errorf("psi contains non-finite value")
			}
		}
	}
	return nil
}

// DeltaStar evaluates the cylindrical Grad-Shafranov operator on psi.
func DeltaStar(c Case, psi [][]float64) ([][]float64, error) {
	if err := validateFluxMatrix(c, psi); err != nil {
		return nil, err
	}
	r, _, _, dR, dZ := grid(c)
	deltaStar := zeros(c.NZ, c.NR)
	dR2 := dR * dR
	dZ2 := dZ * dZ
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			d2DR2 := (psi[iz][ir+1] - 2*psi[iz][ir] + psi[iz][ir-1]) / dR2
			dDROverR := (psi[iz][ir+1] - psi[iz][ir-1]) / (2 * dR * r[ir])
			d2DZ2 := (psi[iz+1][ir] - 2*psi[iz][ir] + psi[iz-1][ir]) / dZ2
			deltaStar[iz][ir] = d2DR2 - dDROverR + d2DZ2
		}
	}
	return deltaStar, nil
}

// ToroidalCurrentDensityFromFlux returns J_phi implied by
// Delta-star psi = -mu0 R J_phi.
func ToroidalCurrentDensityFromFlux(c Case, psi [][]float64) ([][]float64, error) {
	if err := validateFluxMatrix(c, psi); err != nil {
		return nil, err
	}
	r, _, _, _, _ := grid(c)
	deltaStar, err := DeltaStar(c, psi)
	if err != nil {
		return nil, err
	}
	currentDensity := zeros(c.NZ, c.NR)
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			currentDensity[iz][ir] = -deltaStar[iz][ir] / (c.Mu0 * r[ir])
		}
	}
	return currentDensity, nil
}

// TotalToroidalCurrentFromFlux integrates interior J_phi with rectangular
// grid weights.
func TotalToroidalCurrentFromFlux(c Case, psi [][]float64) (float64, error) {
	currentDensity, err := ToroidalCurrentDensityFromFlux(c, psi)
	if err != nil {
		return 0, err
	}
	_, _, _, dR, dZ := grid(c)
	total := 0.0
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			total += currentDensity[iz][ir] * dR * dZ
		}
	}
	if math.IsNaN(total) || math.IsInf(total, 0) {
		return 0, fmt.Errorf("integrated toroidal current became non-finite")
	}
	return total, nil
}

// TotalToroidalCurrentFromFluxTrapezoidal integrates J_phi with full-domain
// trapezoidal weights.
func TotalToroidalCurrentFromFluxTrapezoidal(c Case, psi [][]float64) (float64, error) {
	currentDensity, err := ToroidalCurrentDensityFromFlux(c, psi)
	if err != nil {
		return 0, err
	}
	_, _, _, dR, dZ := grid(c)
	total := 0.0
	for iz := 0; iz < c.NZ; iz++ {
		zWeight := 1.0
		if iz == 0 || iz == c.NZ-1 {
			zWeight = 0.5
		}
		for ir := 0; ir < c.NR; ir++ {
			rWeight := 1.0
			if ir == 0 || ir == c.NR-1 {
				rWeight = 0.5
			}
			total += currentDensity[iz][ir] * zWeight * rWeight * dR * dZ
		}
	}
	if math.IsNaN(total) || math.IsInf(total, 0) {
		return 0, fmt.Errorf("trapezoidal integrated toroidal current became non-finite")
	}
	return total, nil
}

// TotalToroidalCurrentFromFluxMasked integrates J_phi only where domainMask is
// true and rejects empty or shape-incompatible masks.
func TotalToroidalCurrentFromFluxMasked(c Case, psi [][]float64, domainMask [][]bool) (float64, error) {
	if err := validateFluxMatrix(c, psi); err != nil {
		return 0, err
	}
	if len(domainMask) != c.NZ {
		return 0, fmt.Errorf("toroidal current mask row count mismatch")
	}
	hasDomainCell := false
	for _, row := range domainMask {
		if len(row) != c.NR {
			return 0, fmt.Errorf("toroidal current mask column count mismatch")
		}
		for _, inDomain := range row {
			hasDomainCell = hasDomainCell || inDomain
		}
	}
	if !hasDomainCell {
		return 0, fmt.Errorf("toroidal current mask must include at least one cell")
	}
	currentDensity, err := ToroidalCurrentDensityFromFlux(c, psi)
	if err != nil {
		return 0, err
	}
	_, _, _, dR, dZ := grid(c)
	total := 0.0
	for iz := 0; iz < c.NZ; iz++ {
		for ir := 0; ir < c.NR; ir++ {
			if domainMask[iz][ir] {
				total += currentDensity[iz][ir] * dR * dZ
			}
		}
	}
	if math.IsNaN(total) || math.IsInf(total, 0) {
		return 0, fmt.Errorf("masked integrated toroidal current became non-finite")
	}
	return total, nil
}
