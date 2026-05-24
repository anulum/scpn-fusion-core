// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Native Go Grad-Shafranov Solver
package gssolver

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type Case struct {
	RMin, RMax, ZMin, ZMax float64
	NR, NZ                 int
	IpTarget, Mu0          float64
	NPicard, NJacobi       int
	Alpha, OmegaJ, BetaMix float64
}

type Result struct {
	Psi             [][]float64
	ResidualHistory []float64
}

func CaseFromTOML(path string) (Case, error) {
	file, err := os.Open(path)
	if err != nil {
		return Case{}, err
	}
	defer file.Close()

	values := map[string]string{}
	section := ""
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(strings.Split(scanner.Text(), "#")[0])
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			section = strings.TrimSuffix(strings.TrimPrefix(line, "["), "]")
			continue
		}
		if section != "grad_shafranov" || !strings.Contains(line, "=") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		values[strings.TrimSpace(parts[0])] = strings.Trim(strings.TrimSpace(parts[1]), `"`)
	}
	if err := scanner.Err(); err != nil {
		return Case{}, err
	}

	parseFloat := func(key string) (float64, error) {
		value, ok := values[key]
		if !ok {
			return 0, fmt.Errorf("missing %s", key)
		}
		return strconv.ParseFloat(value, 64)
	}
	parseInt := func(key string) (int, error) {
		value, ok := values[key]
		if !ok {
			return 0, fmt.Errorf("missing %s", key)
		}
		parsed, err := strconv.Atoi(value)
		return parsed, err
	}

	var c Case
	if c.RMin, err = parseFloat("R_min"); err != nil {
		return Case{}, err
	}
	if c.RMax, err = parseFloat("R_max"); err != nil {
		return Case{}, err
	}
	if c.ZMin, err = parseFloat("Z_min"); err != nil {
		return Case{}, err
	}
	if c.ZMax, err = parseFloat("Z_max"); err != nil {
		return Case{}, err
	}
	if c.NR, err = parseInt("NR"); err != nil {
		return Case{}, err
	}
	if c.NZ, err = parseInt("NZ"); err != nil {
		return Case{}, err
	}
	if c.IpTarget, err = parseFloat("Ip_target"); err != nil {
		return Case{}, err
	}
	if c.Mu0, err = parseFloat("mu0"); err != nil {
		return Case{}, err
	}
	if c.NPicard, err = parseInt("n_picard"); err != nil {
		return Case{}, err
	}
	if c.NJacobi, err = parseInt("n_jacobi"); err != nil {
		return Case{}, err
	}
	if c.Alpha, err = parseFloat("alpha"); err != nil {
		return Case{}, err
	}
	if c.OmegaJ, err = parseFloat("omega_j"); err != nil {
		return Case{}, err
	}
	if c.BetaMix, err = parseFloat("beta_mix"); err != nil {
		return Case{}, err
	}
	return c, c.Validate()
}

func (c Case) Validate() error {
	if !(c.RMax > c.RMin) || !(c.ZMax > c.ZMin) {
		return fmt.Errorf("invalid domain bounds")
	}
	if c.NR < 3 || c.NZ < 3 {
		return fmt.Errorf("grid dimensions must be at least 3")
	}
	if c.Mu0 <= 0 || c.NPicard < 1 || c.NJacobi < 1 {
		return fmt.Errorf("invalid positive solver scalar")
	}
	if !(c.Alpha > 0 && c.Alpha <= 1) || !(c.OmegaJ > 0 && c.OmegaJ < 2) || !(c.BetaMix >= 0 && c.BetaMix <= 1) {
		return fmt.Errorf("invalid relaxation or profile scalar")
	}
	for _, value := range []float64{c.RMin, c.RMax, c.ZMin, c.ZMax, c.IpTarget, c.Mu0, c.Alpha, c.OmegaJ, c.BetaMix} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("non-finite case scalar")
		}
	}
	return nil
}

func Solve(c Case) (Result, error) {
	if err := c.Validate(); err != nil {
		return Result{}, err
	}
	r, _, rr, dR, dZ := grid(c)
	_ = r
	psi := initialPsi(c, rr)
	residuals := make([]float64, 0, c.NPicard)
	for outer := 0; outer < c.NPicard; outer++ {
		source := computeSource(c, psi, rr, dR, dZ)
		psiElliptic := clone(psi)
		for inner := 0; inner < c.NJacobi; inner++ {
			psiElliptic = jacobiStep(c, psiElliptic, source, rr, dR, dZ)
		}
		psiNext := zeros(c.NZ, c.NR)
		maxChange := 0.0
		for iz := 0; iz < c.NZ; iz++ {
			for ir := 0; ir < c.NR; ir++ {
				psiNext[iz][ir] = (1-c.Alpha)*psi[iz][ir] + c.Alpha*psiElliptic[iz][ir]
				maxChange = math.Max(maxChange, math.Abs(psiNext[iz][ir]-psi[iz][ir]))
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

func initialPsi(c Case, rr [][]float64) [][]float64 {
	psi := zeros(c.NZ, c.NR)
	rCenter := 0.5 * (c.RMin + c.RMax)
	for iz := range psi {
		for ir := range psi[iz] {
			delta := rr[iz][ir] - rCenter
			psi[iz][ir] = math.Exp(-(delta*delta)/0.5) * 0.01
		}
	}
	applyZeroBoundary(psi)
	return psi
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

func computeSource(c Case, psi, rr [][]float64, dR, dZ float64) [][]float64 {
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
			psiNorm = math.Min(1, math.Max(0, psiNorm))
			profile := 0.0
			if psiNorm >= 0 && psiNorm < 1 {
				profile = 1 - psiNorm
			}
			rSafe := math.Max(rr[iz][ir], 1e-10)
			jP := rr[iz][ir] * profile
			jF := profile / (c.Mu0 * rSafe)
			jRaw[iz][ir] = c.BetaMix*jP + (1-c.BetaMix)*jF
			current += jRaw[iz][ir] * dR * dZ
		}
	}
	scale := c.IpTarget / math.Max(math.Abs(current), 1e-9)
	source := zeros(c.NZ, c.NR)
	for iz := range source {
		for ir := range source[iz] {
			source[iz][ir] = -c.Mu0 * rr[iz][ir] * jRaw[iz][ir] * scale
		}
	}
	return source
}

func jacobiStep(c Case, psi, source, rr [][]float64, dR, dZ float64) [][]float64 {
	out := clone(psi)
	dR2 := dR * dR
	dZ2 := dZ * dZ
	aNS := 1 / dZ2
	aC := 2/dR2 + 2/dZ2
	for iz := 1; iz < c.NZ-1; iz++ {
		for ir := 1; ir < c.NR-1; ir++ {
			rSafe := math.Max(rr[iz][ir], 1e-10)
			aE := 1/dR2 - 1/(2*rSafe*dR)
			aW := 1/dR2 + 1/(2*rSafe*dR)
			update := (aE*psi[iz][ir+1] + aW*psi[iz][ir-1] + aNS*(psi[iz-1][ir]+psi[iz+1][ir]) - source[iz][ir]) / aC
			out[iz][ir] = (1-c.OmegaJ)*psi[iz][ir] + c.OmegaJ*update
		}
	}
	return out
}
