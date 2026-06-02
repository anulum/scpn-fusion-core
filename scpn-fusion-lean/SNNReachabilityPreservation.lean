/-
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SCPN Fusion Core — Lean Petri-to-SNN Reachability Proof
-/
import SCPNFusionSolvers

namespace SCPNFusionSolvers

/-!
Petri-to-SNN reachability contract:

The executable compiler stores Petri input/output incidence as matrices. This
Lean proof models the safety-relevant graph contract: every Petri adjacency
edge emitted by the symbolic net is present in the compiled SNN edge contract,
and therefore every finite Petri reachability path is preserved by compilation.
-/

structure PetriGraph where
  placeCount : Nat
  transitionCount : Nat
  edges : List (Nat × Nat)
  deriving Repr

structure SnnGraph where
  neuronCount : Nat
  edges : List (Nat × Nat)
  deriving Repr

def compilePetriToSnnGraph (net : PetriGraph) : SnnGraph :=
  { neuronCount := net.placeCount + net.transitionCount, edges := net.edges }

def EdgeWithin (nodeCount : Nat) (edge : Nat × Nat) : Prop :=
  edge.fst < nodeCount ∧ edge.snd < nodeCount

def PetriGraph.WellFormed (net : PetriGraph) : Prop :=
  ∀ edge, edge ∈ net.edges → EdgeWithin (net.placeCount + net.transitionCount) edge

inductive PetriReachable (edges : List (Nat × Nat)) : Nat → Nat → Prop where
  | step {src dst : Nat} : (src, dst) ∈ edges → PetriReachable edges src dst
  | trans {src mid dst : Nat} :
      PetriReachable edges src mid →
      PetriReachable edges mid dst →
      PetriReachable edges src dst

inductive SnnReachable (edges : List (Nat × Nat)) : Nat → Nat → Prop where
  | step {src dst : Nat} : (src, dst) ∈ edges → SnnReachable edges src dst
  | trans {src mid dst : Nat} :
      SnnReachable edges src mid →
      SnnReachable edges mid dst →
      SnnReachable edges src dst

theorem petri_reachability_compose
    {edges : List (Nat × Nat)} {src mid dst : Nat}
    (left : PetriReachable edges src mid)
    (right : PetriReachable edges mid dst) :
    PetriReachable edges src dst := by
  exact PetriReachable.trans left right

theorem snn_reachability_compose
    {edges : List (Nat × Nat)} {src mid dst : Nat}
    (left : SnnReachable edges src mid)
    (right : SnnReachable edges mid dst) :
    SnnReachable edges src dst := by
  exact SnnReachable.trans left right

theorem petri_no_reachability_without_edges {src dst : Nat} :
    ¬ PetriReachable [] src dst := by
  intro h
  induction h with
  | step edge =>
      cases edge
  | trans left right leftImpossible rightImpossible =>
      exact leftImpossible

theorem petri_no_direct_edge_without_edges {src dst : Nat} :
    ¬ (src, dst) ∈ ([] : List (Nat × Nat)) := by
  intro edge
  cases edge

theorem snn_no_reachability_without_edges {src dst : Nat} :
    ¬ SnnReachable [] src dst := by
  intro h
  induction h with
  | step edge =>
      cases edge
  | trans left right leftImpossible rightImpossible =>
      exact leftImpossible

theorem snn_no_direct_edge_without_edges {src dst : Nat} :
    ¬ (src, dst) ∈ ([] : List (Nat × Nat)) := by
  intro edge
  cases edge

theorem compile_preserves_neuron_count (net : PetriGraph) :
    (compilePetriToSnnGraph net).neuronCount = net.placeCount + net.transitionCount := by
  rfl

theorem compile_preserves_edge_list (net : PetriGraph) :
    (compilePetriToSnnGraph net).edges = net.edges := by
  rfl

theorem compile_preserves_edge_count (net : PetriGraph) :
    (compilePetriToSnnGraph net).edges.length = net.edges.length := by
  rfl

theorem compile_preserves_empty_edges
    {placeCount transitionCount : Nat} :
    (compilePetriToSnnGraph
      { placeCount := placeCount,
        transitionCount := transitionCount,
        edges := [] }).edges = [] := by
  rfl

theorem compile_empty_edge_count_zero
    {placeCount transitionCount : Nat} :
    (compilePetriToSnnGraph
      { placeCount := placeCount,
        transitionCount := transitionCount,
        edges := [] }).edges.length = 0 := by
  rfl

theorem compile_preserves_direct_edge
    (net : PetriGraph) {src dst : Nat}
    (h : (src, dst) ∈ net.edges) :
    (src, dst) ∈ (compilePetriToSnnGraph net).edges := by
  exact h

theorem compile_reflects_direct_edge
    (net : PetriGraph) {src dst : Nat}
    (h : (src, dst) ∈ (compilePetriToSnnGraph net).edges) :
    (src, dst) ∈ net.edges := by
  exact h

theorem compile_direct_edge_equivalent
    (net : PetriGraph) {src dst : Nat} :
    (src, dst) ∈ net.edges ↔ (src, dst) ∈ (compilePetriToSnnGraph net).edges := by
  constructor
  · intro h
    exact compile_preserves_direct_edge net h
  · intro h
    exact compile_reflects_direct_edge net h

theorem compile_preserves_well_formed_edges
    (net : PetriGraph) (h : net.WellFormed) :
    ∀ edge, edge ∈ (compilePetriToSnnGraph net).edges →
      EdgeWithin (compilePetriToSnnGraph net).neuronCount edge := by
  intro edge edgeInCompiled
  exact h edge edgeInCompiled

theorem compile_direct_edge_source_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : (src, dst) ∈ (compilePetriToSnnGraph net).edges) :
    src < (compilePetriToSnnGraph net).neuronCount := by
  exact (compile_preserves_well_formed_edges net wellFormed (src, dst) h).left

theorem compile_direct_edge_destination_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : (src, dst) ∈ (compilePetriToSnnGraph net).edges) :
    dst < (compilePetriToSnnGraph net).neuronCount := by
  exact (compile_preserves_well_formed_edges net wellFormed (src, dst) h).right

theorem petri_direct_edge_source_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : (src, dst) ∈ net.edges) :
    src < net.placeCount + net.transitionCount := by
  exact (wellFormed (src, dst) h).left

theorem petri_direct_edge_destination_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : (src, dst) ∈ net.edges) :
    dst < net.placeCount + net.transitionCount := by
  exact (wellFormed (src, dst) h).right

theorem compile_well_formed_iff
    (net : PetriGraph) :
    (∀ edge, edge ∈ (compilePetriToSnnGraph net).edges →
      EdgeWithin (compilePetriToSnnGraph net).neuronCount edge) ↔
      net.WellFormed := by
  constructor
  · intro h edge edgeInPetri
    exact h edge edgeInPetri
  · intro h edge edgeInCompiled
    exact h edge edgeInCompiled

theorem compile_preserves_reachability
    (net : PetriGraph) {src dst : Nat}
    (h : PetriReachable net.edges src dst) :
    SnnReachable (compilePetriToSnnGraph net).edges src dst := by
  induction h with
  | step edge =>
      exact SnnReachable.step edge
  | trans left right leftPreserved rightPreserved =>
      exact SnnReachable.trans leftPreserved rightPreserved

theorem compile_reflects_reachability
    (net : PetriGraph) {src dst : Nat}
    (h : SnnReachable (compilePetriToSnnGraph net).edges src dst) :
    PetriReachable net.edges src dst := by
  induction h with
  | step edge =>
      exact PetriReachable.step edge
  | trans left right leftReflected rightReflected =>
      exact PetriReachable.trans leftReflected rightReflected

theorem compile_reachability_equivalent
    (net : PetriGraph) {src dst : Nat} :
    PetriReachable net.edges src dst ↔
      SnnReachable (compilePetriToSnnGraph net).edges src dst := by
  constructor
  · intro h
    exact compile_preserves_reachability net h
  · intro h
    exact compile_reflects_reachability net h

theorem petri_reachability_endpoints_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : PetriReachable net.edges src dst) :
    src < net.placeCount + net.transitionCount ∧
      dst < net.placeCount + net.transitionCount := by
  induction h with
  | step edge =>
      exact wellFormed _ edge
  | trans left right leftWithin rightWithin =>
      exact And.intro leftWithin.left rightWithin.right

theorem petri_reachability_source_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : PetriReachable net.edges src dst) :
    src < net.placeCount + net.transitionCount := by
  exact (petri_reachability_endpoints_within net wellFormed h).left

theorem petri_reachability_destination_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : PetriReachable net.edges src dst) :
    dst < net.placeCount + net.transitionCount := by
  exact (petri_reachability_endpoints_within net wellFormed h).right

theorem snn_reachability_endpoints_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : SnnReachable (compilePetriToSnnGraph net).edges src dst) :
    src < (compilePetriToSnnGraph net).neuronCount ∧
      dst < (compilePetriToSnnGraph net).neuronCount := by
  exact petri_reachability_endpoints_within net wellFormed
    (compile_reflects_reachability net h)

theorem snn_reachability_source_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : SnnReachable (compilePetriToSnnGraph net).edges src dst) :
    src < (compilePetriToSnnGraph net).neuronCount := by
  exact (snn_reachability_endpoints_within net wellFormed h).left

theorem snn_reachability_destination_within
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : SnnReachable (compilePetriToSnnGraph net).edges src dst) :
    dst < (compilePetriToSnnGraph net).neuronCount := by
  exact (snn_reachability_endpoints_within net wellFormed h).right

theorem compile_has_no_spurious_reachable_path
    (net : PetriGraph) {src dst : Nat}
    (h : SnnReachable (compilePetriToSnnGraph net).edges src dst) :
    PetriReachable net.edges src dst := by
  exact compile_reflects_reachability net h

theorem compile_reachability_preserves_and_reflects_bounded_endpoints
    (net : PetriGraph) (wellFormed : net.WellFormed)
    {src dst : Nat}
    (h : PetriReachable net.edges src dst) :
    SnnReachable (compilePetriToSnnGraph net).edges src dst ∧
      src < (compilePetriToSnnGraph net).neuronCount ∧
      dst < (compilePetriToSnnGraph net).neuronCount := by
  constructor
  · exact compile_preserves_reachability net h
  · exact petri_reachability_endpoints_within net wellFormed h

theorem compile_preserves_composed_reachability
    (net : PetriGraph) {src mid dst : Nat}
    (left : PetriReachable net.edges src mid)
    (right : PetriReachable net.edges mid dst) :
    SnnReachable (compilePetriToSnnGraph net).edges src dst := by
  exact compile_preserves_reachability net (petri_reachability_compose left right)

theorem compile_reflects_composed_reachability
    (net : PetriGraph) {src mid dst : Nat}
    (left : SnnReachable (compilePetriToSnnGraph net).edges src mid)
    (right : SnnReachable (compilePetriToSnnGraph net).edges mid dst) :
    PetriReachable net.edges src dst := by
  exact compile_reflects_reachability net (snn_reachability_compose left right)

theorem empty_petri_graph_well_formed
    (placeCount transitionCount : Nat) :
    ({ placeCount := placeCount,
       transitionCount := transitionCount,
       edges := [] } : PetriGraph).WellFormed := by
  intro edge edgeInEmpty
  cases edgeInEmpty

theorem compile_empty_graph_has_no_direct_edge
    {placeCount transitionCount src dst : Nat} :
    ¬ (src, dst) ∈
      (compilePetriToSnnGraph
        { placeCount := placeCount,
          transitionCount := transitionCount,
          edges := [] }).edges := by
  exact snn_no_direct_edge_without_edges

theorem compile_empty_graph_has_no_snn_reachability
    {placeCount transitionCount src dst : Nat} :
    ¬ SnnReachable
      (compilePetriToSnnGraph
        { placeCount := placeCount,
          transitionCount := transitionCount,
          edges := [] }).edges src dst := by
  exact snn_no_reachability_without_edges

end SCPNFusionSolvers
