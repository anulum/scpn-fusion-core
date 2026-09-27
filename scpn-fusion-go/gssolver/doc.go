// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li

// Package gssolver provides the native Go fixed-boundary Grad-Shafranov
// reference solver and flux-derived current diagnostics.
// CaseFromTOML reads the exact thirteen-field grad_shafranov TOML 1.0 schema;
// Case.Validate applies the same domain, generated-axis and work limits to
// direct callers before allocation. Counts are signed TOML integers and
// physical fields use SI units. See docs/NUMERICAL_CONTRACTS.md in the repository.
package gssolver
