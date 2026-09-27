// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Go physical case admission tests
package gssolver

import (
	"context"
	"crypto/sha256"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/BurntSushi/toml"
)

// TestSharedPhysicalCaseCorpus verifies pinned bytes and real public loader semantics.
func TestSharedPhysicalCaseCorpus(t *testing.T) {
	folder := "../../validation/polyglot/case_contract"
	var manifest struct {
		Schema string
		Case   []struct {
			Name, Path, SHA256, Reason string
			Valid, Solve               bool
			Expected                   map[string]any
		}
	}
	if _, err := toml.DecodeFile(filepath.Join(folder, "manifest.toml"), &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Schema != "scpn.physical-case-corpus.v1" || len(manifest.Case) == 0 {
		t.Fatal("invalid corpus metadata")
	}
	for _, fixture := range manifest.Case {
		t.Run(fixture.Name, func(t *testing.T) {
			path := filepath.Join(folder, fixture.Path)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if fmt.Sprintf("%x", sha256.Sum256(data)) != fixture.SHA256 {
				t.Fatal("fixture bytes differ from manifest")
			}
			got, err := CaseFromTOML(path)
			if !fixture.Valid {
				if err == nil {
					t.Fatalf("invalid case admitted: %s", fixture.Reason)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			e := fixture.Expected
			want := Case{RMin: e["R_min"].(float64), RMax: e["R_max"].(float64), ZMin: e["Z_min"].(float64), ZMax: e["Z_max"].(float64), NR: int(e["NR"].(int64)), NZ: int(e["NZ"].(int64)), IpTarget: e["Ip_target"].(float64), Mu0: e["mu0"].(float64), NPicard: int(e["n_picard"].(int64)), NJacobi: int(e["n_jacobi"].(int64)), Alpha: e["alpha"].(float64), OmegaJ: e["omega_j"].(float64), BetaMix: e["beta_mix"].(float64)}
			if got != want {
				t.Fatalf("physical mapping differs: %+v vs %+v", got, want)
			}
			if err := got.Validate(); err != nil {
				t.Fatal(err)
			}
			if fixture.Solve {
				result, err := Solve(got)
				if err != nil {
					t.Fatal(err)
				}
				if len(result.Psi) != got.NZ || len(result.Psi[0]) != got.NR {
					t.Fatal("public solve output shape differs")
				}
				for _, row := range result.Psi {
					for _, value := range row {
						if !finite(value) {
							t.Fatal("public solve returned nonfinite flux")
						}
					}
				}
			}
		})
	}
}

// caseDocument reads the shared physical reference, without rewriting it.
func caseDocument(t *testing.T) string {
	t.Helper()
	data, err := os.ReadFile("../../validation/polyglot/gs_picard_reference.toml")
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

// loadDocument exercises the actual public file loader on exact supplied bytes.
func loadDocument(t *testing.T, document string) (Case, error) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "case.toml")
	if err := os.WriteFile(path, []byte(document), 0600); err != nil {
		t.Fatal(err)
	}
	return CaseFromTOML(path)
}

// replaceCaseValue changes exactly one field in the shared reference document.
func replaceCaseValue(document, key, value string) string {
	lines := strings.Split(document, "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, key+" = ") {
			lines[i] = key + " = " + value
		}
	}
	return strings.Join(lines, "\n")
}

// TestCaseFromTOMLTypedTree checks conforming spellings and exact delivered values.
func TestCaseFromTOMLTypedTree(t *testing.T) {
	document := caseDocument(t)
	expected, err := loadDocument(t, document)
	if err != nil {
		t.Fatal(err)
	}
	fields := strings.Split(strings.SplitN(document, "[grad_shafranov]\n", 2)[1], "\n")
	dotted := ""
	inline := []string{}
	for _, field := range fields {
		if field == "" {
			continue
		}
		dotted += "grad_shafranov." + field + "\n"
		inline = append(inline, field)
	}
	cases := map[string]string{
		"table":           document,
		"quoted":          strings.ReplaceAll(document, "[grad_shafranov]", "[\"grad_shafranov\"]"),
		"dotted":          dotted,
		"inline":          "grad_shafranov = {" + strings.Join(inline, ",") + "}\n",
		"radix":           replaceCaseValue(replaceCaseValue(document, "NR", "0x11"), "NZ", "0b10001"),
		"underscores":     replaceCaseValue(document, "Ip_target", "1_000_000"),
		"signed_exponent": replaceCaseValue(document, "Ip_target", "+1.0e+6"),
	}
	for name, doc := range cases {
		t.Run(name, func(t *testing.T) {
			got, err := loadDocument(t, doc)
			if err != nil {
				t.Fatal(err)
			}
			if got != expected {
				t.Fatalf("different physical mapping: %+v vs %+v", got, expected)
			}
		})
	}
	for _, current := range []string{"-1000000", "0", "9007199254740994"} {
		if _, err := loadDocument(t, replaceCaseValue(document, "Ip_target", current)); err != nil {
			t.Fatal(err)
		}
	}
}

// TestCaseFromTOMLRejectsMalformed proves syntax, required tree and typed-field refusal.
func TestCaseFromTOMLRejectsMalformed(t *testing.T) {
	document := caseDocument(t)
	invalid := map[string]string{
		"missing_root":       "",
		"wrong_root_kind":    "grad_shafranov = 1\n",
		"same_count_unknown": strings.Replace(document, "R_min =", "unknown =", 1),
		"duplicate":          document + "NR = 17\n",
		"duplicate_table":    document + "[grad_shafranov]\n",
		"unknown_field":      document + "extra = 1\n",
		"unknown_root":       "extra = 1\n" + document,
		"unknown_table":      document + "[other]\nx = 1\n",
		"malformed":          document + "broken line\n",
		"leading_zero":       replaceCaseValue(document, "NR", "017"),
		"utf8":               document + string([]byte{0xff}),
		"inexact_real":       replaceCaseValue(document, "Ip_target", "9007199254740993"),
		"integer_overflow":   replaceCaseValue(document, "Ip_target", "9223372036854775808"),
	}
	for _, key := range []string{"R_min", "R_max", "Z_min", "Z_max", "NR", "NZ", "Ip_target", "mu0", "n_picard", "n_jacobi", "alpha", "omega_j", "beta_mix"} {
		lines := strings.Split(document, "\n")
		kept := []string{}
		for _, line := range lines {
			if !strings.HasPrefix(line, key+" = ") {
				kept = append(kept, line)
			}
		}
		invalid["missing_"+key] = strings.Join(kept, "\n")
		for _, kind := range []string{"true", "\"1#literal\"", "[1]", "{x=1}", "1979-05-27"} {
			invalid[key+"_"+kind] = replaceCaseValue(document, key, kind)
		}
	}
	for _, key := range []string{"NR", "NZ", "n_picard", "n_jacobi"} {
		invalid[key+"_float"] = replaceCaseValue(document, key, "17.0")
		for _, value := range []string{"-1", "0", "9223372036854775807", "10001"} {
			invalid[key+value] = replaceCaseValue(document, key, value)
		}
	}
	for _, key := range []string{"R_min", "R_max", "Z_min", "Z_max", "Ip_target", "mu0", "alpha", "omega_j", "beta_mix"} {
		for _, value := range []string{"nan", "+inf", "-inf"} {
			invalid[key+value] = replaceCaseValue(document, key, value)
		}
	}
	for name, doc := range invalid {
		t.Run(name, func(t *testing.T) {
			if _, err := loadDocument(t, doc); err == nil {
				t.Fatal("invalid document admitted")
			}
		})
	}
	if _, err := CaseFromTOML(filepath.Join(t.TempDir(), "absent.toml")); err == nil {
		t.Fatal("missing file admitted")
	}
	if _, err := CaseFromTOML(t.TempDir()); err == nil {
		t.Fatal("directory admitted as case")
	}
}

// TestCaseAdmissionLimits checks limits without allocating or running a cap-sized solve.
func TestCaseAdmissionLimits(t *testing.T) {
	valid := referenceCase()
	valid.NR, valid.NZ, valid.NPicard, valid.NJacobi = 10, 10, 10000, 100
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	valid.NPicard = 9999
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	valid.NR, valid.NZ, valid.NPicard, valid.NJacobi = 1025, 3, 1, 1
	if err := valid.Validate(); err != nil {
		t.Fatal(err)
	}
	changes := map[string]func(*Case){
		"nonfinite":         func(c *Case) { c.IpTarget = math.Inf(1) },
		"mu0_zero":          func(c *Case) { c.Mu0 = 0 },
		"alpha_zero":        func(c *Case) { c.Alpha = 0 },
		"omega_upper":       func(c *Case) { c.OmegaJ = 2 },
		"beta_upper":        func(c *Case) { c.BetaMix = math.Nextafter(1, 2) },
		"grid_cap":          func(c *Case) { c.NR = 1026 },
		"iteration_cap":     func(c *Case) { c.NJacobi = 10001 },
		"product":           func(c *Case) { c.NR, c.NZ, c.NPicard, c.NJacobi = 10, 10, 10000, 101 },
		"overflow_count":    func(c *Case) { c.NR = int(^uint(0) >> 1) },
		"positive_radius":   func(c *Case) { c.RMin = 0 },
		"spacing_overflow":  func(c *Case) { c.ZMin, c.ZMax = -math.MaxFloat64, math.MaxFloat64 },
		"spacing_underflow": func(c *Case) { c.ZMin, c.ZMax = 0, math.SmallestNonzeroFloat64 },
		"collapsed_axis":    func(c *Case) { c.RMin, c.RMax = 1, math.Nextafter(1, 2) },
	}
	for name, change := range changes {
		t.Run(name, func(t *testing.T) {
			c := referenceCase()
			change(&c)
			if err := c.Validate(); err == nil {
				t.Fatal("direct Case admitted")
			}
			if _, err := Solve(c); err == nil {
				t.Fatal("direct Solve admitted")
			}
		})
	}
}

// TestCaseCSVCLI exercises the built public CLI, including no CSV on typed failure.
func TestCaseCSVCLI(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	binary := filepath.Join(t.TempDir(), "gs_picard_csv")
	if data, err := exec.CommandContext(ctx, "go", "build", "-o", binary, "../cmd/gs_picard_csv").CombinedOutput(); err != nil {
		t.Fatalf("build: %v: %s", err, data)
	}
	document := caseDocument(t)
	path := filepath.Join(t.TempDir(), "case.toml")
	if err := os.WriteFile(path, []byte(document), 0600); err != nil {
		t.Fatal(err)
	}
	output, err := exec.CommandContext(ctx, binary, path).Output()
	if err != nil {
		t.Fatal(err)
	}
	rows, err := csv.NewReader(strings.NewReader(string(output))).ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	c, err := CaseFromTOML(path)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Solve(c)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != c.NZ {
		t.Fatal("CSV row count mismatch")
	}
	for iz, row := range rows {
		if len(row) != c.NR {
			t.Fatal("CSV column count mismatch")
		}
		for ir, raw := range row {
			got, err := strconv.ParseFloat(raw, 64)
			if err != nil || got != want.Psi[iz][ir] {
				t.Fatalf("CSV physical mapping mismatch at (%d,%d)", iz, ir)
			}
		}
	}
	for name, invalid := range map[string]string{
		"duplicate":   document + "NR = 17\n",
		"count_float": replaceCaseValue(document, "NR", "17.0"),
		"cap":         replaceCaseValue(document, "NR", "1026"),
		"unknown":     document + "extra = 1\n",
		"numerical":   replaceCaseValue(replaceCaseValue(document, "R_min", "1e200"), "R_max", "2e200"),
	} {
		t.Run(name, func(t *testing.T) {
			if err := os.WriteFile(path, []byte(invalid), 0600); err != nil {
				t.Fatal(err)
			}
			out, err := exec.CommandContext(ctx, binary, path).Output()
			failure, ok := err.(*exec.ExitError)
			if !ok || len(out) != 0 || len(failure.Stderr) == 0 || failure.ExitCode() != 1 {
				t.Fatalf("invalid CLI result: stdout=%q error=%v", out, err)
			}
		})
	}
}

// TestSolveRejectsNonfiniteArithmetic checks a real accepted-input numerical failure.
func TestSolveRejectsNonfiniteArithmetic(t *testing.T) {
	c := referenceCase()
	c.RMin, c.RMax = 1e200, 2e200
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := Solve(c); err == nil {
		t.Fatal("nonfinite arithmetic returned successful flux")
	}
}

// TestSolveRejectsHiddenIntermediateOverflow exercises admitted physical inputs.
func TestSolveRejectsHiddenIntermediateOverflow(t *testing.T) {
	for _, item := range []struct {
		name, diagnostic string
		change           func(*Case)
	}{
		{"seed", "initial Gaussian", func(c *Case) { c.RMin, c.RMax = 1e154, 1e155 }},
		{"seed_exponent", "initial Gaussian", func(c *Case) { c.RMin, c.RMax = 1e154, 3e154 }},
		{"scaled_current", "scaled current", func(c *Case) { c.RMin, c.RMax, c.ZMin, c.ZMax, c.IpTarget = 1, 1.01, -0.01, 0.01, 1e308 }},
	} {
		t.Run(item.name, func(t *testing.T) {
			c := referenceCase()
			c.NPicard, c.NJacobi = 1, 1
			item.change(&c)
			if err := c.Validate(); err != nil {
				t.Fatal(err)
			}
			_, err := Solve(c)
			if err == nil || !strings.Contains(err.Error(), item.diagnostic) {
				t.Fatalf("expected %s refusal, got %v", item.diagnostic, err)
			}
		})
	}
}
