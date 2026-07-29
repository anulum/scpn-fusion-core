// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Command doccheck fails when a Go package or exported production declaration
// lacks a native documentation comment.
package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type listedPackage struct {
	// Dir is the absolute package directory returned by go list.
	Dir string
	// ImportPath is the canonical package import path.
	ImportPath string
	// GoFiles contains production Go source filenames.
	GoFiles []string
	// CgoFiles contains production cgo source filenames.
	CgoFiles []string
}

func main() {
	patterns := os.Args[1:]
	if len(patterns) == 0 {
		patterns = []string{"./..."}
	}
	packages, err := listPackages(patterns)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var findings []string
	for _, pkg := range packages {
		findings = append(findings, inspectPackage(pkg)...)
	}
	sort.Strings(findings)
	for _, finding := range findings {
		fmt.Fprintln(os.Stderr, finding)
	}
	if len(findings) != 0 {
		fmt.Fprintf(os.Stderr, "doccheck: %d missing documentation comment(s)\n", len(findings))
		os.Exit(1)
	}
	fmt.Printf("doccheck: %d package(s) documented\n", len(packages))
}

func listPackages(patterns []string) ([]listedPackage, error) {
	args := append([]string{"list", "-json"}, patterns...)
	output, err := exec.Command("go", args...).Output()
	if err != nil {
		return nil, fmt.Errorf("go list: %w", err)
	}
	decoder := json.NewDecoder(strings.NewReader(string(output)))
	var packages []listedPackage
	for decoder.More() {
		var pkg listedPackage
		if err := decoder.Decode(&pkg); err != nil {
			return nil, fmt.Errorf("decode go list output: %w", err)
		}
		packages = append(packages, pkg)
	}
	if len(packages) == 0 {
		return nil, fmt.Errorf("go list returned no packages")
	}
	return packages, nil
}

func inspectPackage(pkg listedPackage) []string {
	files := append(append([]string{}, pkg.GoFiles...), pkg.CgoFiles...)
	fileSet := token.NewFileSet()
	var findings []string
	hasPackageDoc := false
	for _, name := range files {
		path := filepath.Join(pkg.Dir, name)
		file, err := parser.ParseFile(fileSet, path, nil, parser.ParseComments)
		if err != nil {
			findings = append(findings, fmt.Sprintf("%s: parse: %v", pkg.ImportPath, err))
			continue
		}
		if file.Doc != nil && strings.TrimSpace(file.Doc.Text()) != "" {
			hasPackageDoc = true
		}
		findings = append(findings, inspectFile(fileSet, file, pkg.ImportPath)...)
	}
	if !hasPackageDoc {
		findings = append(findings, fmt.Sprintf("%s: package comment", pkg.ImportPath))
	}
	return findings
}

func inspectFile(fileSet *token.FileSet, file *ast.File, importPath string) []string {
	var findings []string
	for _, declaration := range file.Decls {
		switch node := declaration.(type) {
		case *ast.FuncDecl:
			if ast.IsExported(node.Name.Name) && node.Doc == nil {
				findings = append(findings, location(fileSet, node.Pos(), importPath, node.Name.Name))
			}
		case *ast.GenDecl:
			for _, specification := range node.Specs {
				switch spec := specification.(type) {
				case *ast.TypeSpec:
					if ast.IsExported(spec.Name.Name) && spec.Doc == nil && node.Doc == nil {
						findings = append(findings, location(fileSet, spec.Pos(), importPath, spec.Name.Name))
					}
					if structure, ok := spec.Type.(*ast.StructType); ok {
						findings = append(findings, inspectFields(fileSet, structure.Fields, importPath, spec.Name.Name)...)
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						if ast.IsExported(name.Name) && spec.Doc == nil && node.Doc == nil {
							findings = append(findings, location(fileSet, name.Pos(), importPath, name.Name))
						}
					}
				}
			}
		}
	}
	return findings
}

func inspectFields(fileSet *token.FileSet, fields *ast.FieldList, importPath string, owner string) []string {
	var findings []string
	for _, field := range fields.List {
		for _, name := range field.Names {
			if ast.IsExported(name.Name) && field.Doc == nil && field.Comment == nil {
				findings = append(findings, location(fileSet, name.Pos(), importPath, owner+"."+name.Name))
			}
		}
	}
	return findings
}

func location(fileSet *token.FileSet, position token.Pos, importPath string, name string) string {
	pos := fileSet.Position(position)
	return fmt.Sprintf("%s:%s:%d: %s", importPath, filepath.Base(pos.Filename), pos.Line, name)
}
