// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Fusion Core — Go Grad-Shafranov CSV CLI
package main

import (
	"fmt"
	"os"

	"anulum.li/scpn-fusion-go/gssolver"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: gs_picard_csv CASE.toml")
		os.Exit(2)
	}
	c, err := gssolver.CaseFromTOML(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	result, err := gssolver.Solve(c)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for iz, row := range result.Psi {
		for ir, value := range row {
			if ir > 0 {
				fmt.Print(",")
			}
			fmt.Printf("%.17g", value)
		}
		if iz+1 < len(result.Psi) {
			fmt.Println()
		}
	}
}
