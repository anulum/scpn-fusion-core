// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Typed TOML physical case validation
package gssolver

import (
	"fmt"
	"math"
	"math/big"
	"os"
	"unicode/utf8"

	"github.com/BurntSushi/toml"
)

// CaseFromTOML parses TOML 1.0 and admits exactly the required physical case tree.
// All thirteen fields are mandatory; unknown entries and numeric coercion fail.
// Radii and heights use metres, current uses amperes and Mu0 uses H/m.
func CaseFromTOML(path string) (Case, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Case{}, err
	}
	if !utf8.Valid(data) {
		return Case{}, fmt.Errorf("case TOML must be valid UTF-8")
	}
	var root map[string]any
	if _, err := toml.Decode(string(data), &root); err != nil {
		return Case{}, err
	}
	fields, ok := root["grad_shafranov"].(map[string]any)
	if len(root) != 1 || !ok {
		return Case{}, fmt.Errorf("expected only the grad_shafranov table")
	}
	var c Case
	reals := []struct {
		key string
		out *float64
	}{
		{"R_min", &c.RMin}, {"R_max", &c.RMax}, {"Z_min", &c.ZMin}, {"Z_max", &c.ZMax},
		{"Ip_target", &c.IpTarget}, {"mu0", &c.Mu0}, {"alpha", &c.Alpha},
		{"omega_j", &c.OmegaJ}, {"beta_mix", &c.BetaMix},
	}
	counts := []struct {
		key string
		out *int
	}{
		{"NR", &c.NR}, {"NZ", &c.NZ}, {"n_picard", &c.NPicard}, {"n_jacobi", &c.NJacobi},
	}
	for _, field := range reals {
		if _, exists := fields[field.key]; !exists {
			return Case{}, fmt.Errorf("missing required Grad-Shafranov case field: %s", field.key)
		}
	}
	for _, field := range counts {
		if _, exists := fields[field.key]; !exists {
			return Case{}, fmt.Errorf("missing required Grad-Shafranov case field: %s", field.key)
		}
	}
	if len(fields) != len(reals)+len(counts) {
		return Case{}, fmt.Errorf("expected exactly thirteen case fields")
	}
	for _, field := range reals {
		value := fields[field.key]
		parsed, err := caseReal(value)
		if err != nil {
			return Case{}, fmt.Errorf("%s: %w", field.key, err)
		}
		*field.out = parsed
	}
	for _, field := range counts {
		value, ok := fields[field.key].(int64)
		if !ok {
			return Case{}, fmt.Errorf("%s must be a TOML integer", field.key)
		}
		if value < 1 || uint64(value) > uint64(^uint(0)>>1) {
			return Case{}, fmt.Errorf("%s is outside the integer domain", field.key)
		}
		*field.out = int(value)
	}
	if err := c.Validate(); err != nil {
		return Case{}, err
	}
	return c, nil
}

// caseReal accepts only finite TOML floats and mathematically exact integer conversions.
func caseReal(value any) (float64, error) {
	var out float64
	switch number := value.(type) {
	case float64:
		out = number
	case int64:
		out = float64(number)
		converted, _ := new(big.Float).SetFloat64(out).Int(nil)
		if converted.Cmp(big.NewInt(number)) != 0 {
			return 0, fmt.Errorf("integer is not exactly representable in binary64")
		}
	default:
		return 0, fmt.Errorf("expected a TOML float or exactly representable integer")
	}
	if !finite(out) {
		return 0, fmt.Errorf("real scalar must be finite")
	}
	return out, nil
}

// Validate checks the physical domain and software admission limits before allocation.
// Limits are 1025 points per axis, 10000 iterations of either kind and 100000000
// point iterations in total. They do not qualify accuracy or authorise that workload.
func (c Case) Validate() error {
	for _, value := range []float64{c.RMin, c.RMax, c.ZMin, c.ZMax, c.IpTarget, c.Mu0, c.Alpha, c.OmegaJ, c.BetaMix} {
		if !finite(value) {
			return fmt.Errorf("non-finite case scalar")
		}
	}
	if !(c.RMin > 0 && c.RMax > c.RMin && c.ZMax > c.ZMin) {
		return fmt.Errorf("invalid domain bounds: require positive ordered R and ordered Z")
	}
	if c.NR < 3 || c.NZ < 3 || c.NR > 1025 || c.NZ > 1025 {
		return fmt.Errorf("grid dimensions must be in [3,1025]")
	}
	if c.NPicard < 1 || c.NJacobi < 1 || c.NPicard > 10000 || c.NJacobi > 10000 {
		return fmt.Errorf("iteration counts must be in [1,10000]")
	}
	work := int64(1)
	for _, count := range []int{c.NR, c.NZ, c.NPicard, c.NJacobi} {
		if int64(count) > 100000000/work {
			return fmt.Errorf("case exceeds 100000000 point iterations")
		}
		work *= int64(count)
	}
	if c.Mu0 <= 0 {
		return fmt.Errorf("mu0 must be positive")
	}
	if !(c.Alpha > 0 && c.Alpha <= 1 && c.OmegaJ > 0 && c.OmegaJ < 2 && c.BetaMix >= 0 && c.BetaMix <= 1) {
		return fmt.Errorf("invalid relaxation or profile scalar")
	}
	for _, axis := range []struct {
		start, stop float64
		count       int
	}{{c.RMin, c.RMax, c.NR}, {c.ZMin, c.ZMax, c.NZ}} {
		step := (axis.stop - axis.start) / float64(axis.count-1)
		if !finite(step) || step <= 0 {
			return fmt.Errorf("grid spacing must be finite and positive")
		}
		if axis.stop <= axis.start+step*float64(axis.count-2) {
			return fmt.Errorf("inclusive grid endpoint has collapsed adjacent nodes")
		}
		previous := axis.start
		for i := 1; i < axis.count; i++ {
			current := axis.start + step*float64(i)
			if !finite(current) || current <= previous {
				return fmt.Errorf("grid axis is non-finite or has collapsed adjacent nodes")
			}
			previous = current
		}
	}
	return nil
}

// finite reports whether a binary64 value is neither NaN nor infinite.
func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
