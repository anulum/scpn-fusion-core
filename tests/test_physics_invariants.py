"""Tests for P2.1: Physics Invariant contracts."""

import math

import pytest

from scpn_fusion.scpn.contracts import (
    DEFAULT_PHYSICS_INVARIANTS,
    PhysicsInvariant,
    PhysicsInvariantViolation,
    check_all_invariants,
    check_physics_invariant,
    should_trigger_mitigation,
)


# ── PhysicsInvariant dataclass ─────────────────────────────────────────


class TestPhysicsInvariantDataclass:
    def test_valid_comparators(self):
        for comp in ("gt", "lt", "gte", "lte"):
            inv = PhysicsInvariant(
                name="test", description="d", threshold=1.0, comparator=comp
            )
            assert inv.comparator == comp

    def test_invalid_comparator_raises(self):
        with pytest.raises(ValueError, match="Invalid comparator"):
            PhysicsInvariant(
                name="test", description="d", threshold=1.0, comparator="eq"
            )

    def test_nonfinite_threshold_raises(self):
        with pytest.raises(ValueError, match="finite"):
            PhysicsInvariant(
                name="test", description="d", threshold=float("inf"), comparator="gt"
            )

    def test_frozen(self):
        inv = PhysicsInvariant(
            name="test", description="d", threshold=1.0, comparator="gt"
        )
        with pytest.raises(AttributeError):
            inv.threshold = 2.0  # type: ignore[misc]


# ── Default invariants ─────────────────────────────────────────────────


class TestDefaultInvariants:
    def test_five_defaults(self):
        assert len(DEFAULT_PHYSICS_INVARIANTS) == 5

    def test_default_names(self):
        names = {inv.name for inv in DEFAULT_PHYSICS_INVARIANTS}
        assert names == {"q_min", "beta_N", "greenwald", "T_i", "energy_conservation_error"}

    def test_q_min_invariant(self):
        inv = next(i for i in DEFAULT_PHYSICS_INVARIANTS if i.name == "q_min")
        assert inv.threshold == 1.0
        assert inv.comparator == "gt"

    def test_beta_N_invariant(self):
        inv = next(i for i in DEFAULT_PHYSICS_INVARIANTS if i.name == "beta_N")
        assert inv.threshold == 2.8
        assert inv.comparator == "lt"

    def test_greenwald_invariant(self):
        inv = next(i for i in DEFAULT_PHYSICS_INVARIANTS if i.name == "greenwald")
        assert inv.threshold == 1.2
        assert inv.comparator == "lt"


# ── check_physics_invariant ────────────────────────────────────────────


class TestCheckPhysicsInvariant:
    def test_satisfied_gt(self):
        inv = PhysicsInvariant(name="q", description="", threshold=1.0, comparator="gt")
        assert check_physics_invariant(inv, 1.5) is None

    def test_violated_gt(self):
        inv = PhysicsInvariant(name="q", description="", threshold=1.0, comparator="gt")
        result = check_physics_invariant(inv, 0.8)
        assert result is not None
        assert result.severity in ("warning", "critical")
        assert result.margin == pytest.approx(0.2, abs=1e-10)

    def test_satisfied_lt(self):
        inv = PhysicsInvariant(name="b", description="", threshold=2.8, comparator="lt")
        assert check_physics_invariant(inv, 2.0) is None

    def test_violated_lt_warning(self):
        inv = PhysicsInvariant(name="b", description="", threshold=2.8, comparator="lt")
        # 3.0 is 0.2 above threshold. 20% of 2.8 = 0.56. 0.2 < 0.56 -> warning
        result = check_physics_invariant(inv, 3.0)
        assert result is not None
        assert result.severity == "warning"

    def test_violated_lt_critical(self):
        inv = PhysicsInvariant(name="b", description="", threshold=2.8, comparator="lt")
        # 5.0 is 2.2 above threshold. 20% of 2.8 = 0.56. 2.2 > 0.56 -> critical
        result = check_physics_invariant(inv, 5.0)
        assert result is not None
        assert result.severity == "critical"

    def test_nan_is_critical(self):
        inv = PhysicsInvariant(name="q", description="", threshold=1.0, comparator="gt")
        result = check_physics_invariant(inv, float("nan"))
        assert result is not None
        assert result.severity == "critical"

    def test_inf_is_critical(self):
        inv = PhysicsInvariant(name="q", description="", threshold=1.0, comparator="gt")
        result = check_physics_invariant(inv, float("inf"))
        assert result is not None
        assert result.severity == "critical"

    def test_satisfied_gte(self):
        inv = PhysicsInvariant(name="x", description="", threshold=1.0, comparator="gte")
        assert check_physics_invariant(inv, 1.0) is None

    def test_satisfied_lte(self):
        inv = PhysicsInvariant(name="x", description="", threshold=1.0, comparator="lte")
        assert check_physics_invariant(inv, 1.0) is None

    def test_boundary_gt_exactly_at_threshold(self):
        inv = PhysicsInvariant(name="q", description="", threshold=1.0, comparator="gt")
        # Exactly 1.0 does NOT satisfy "gt 1.0"
        result = check_physics_invariant(inv, 1.0)
        assert result is not None


# ── check_all_invariants ───────────────────────────────────────────────


class TestCheckAllInvariants:
    def test_all_nominal(self):
        values = {
            "q_min": 1.5,
            "beta_N": 2.0,
            "greenwald": 0.8,
            "T_i": 15.0,
            "energy_conservation_error": 0.005,
        }
        violations = check_all_invariants(values)
        assert len(violations) == 0

    def test_one_violation(self):
        values = {
            "q_min": 0.5,  # violated
            "beta_N": 2.0,
            "greenwald": 0.8,
            "T_i": 15.0,
            "energy_conservation_error": 0.005,
        }
        violations = check_all_invariants(values)
        assert len(violations) == 1
        assert violations[0].invariant.name == "q_min"

    def test_multiple_violations(self):
        values = {
            "q_min": 0.5,  # violated
            "beta_N": 5.0,  # violated
            "greenwald": 2.0,  # violated
        }
        violations = check_all_invariants(values)
        assert len(violations) == 3

    def test_missing_keys_ignored(self):
        values = {"q_min": 1.5}  # only one key
        violations = check_all_invariants(values)
        assert len(violations) == 0

    def test_empty_values(self):
        violations = check_all_invariants({})
        assert len(violations) == 0

    def test_custom_invariants(self):
        custom = [
            PhysicsInvariant(name="custom", description="", threshold=10.0, comparator="lt"),
        ]
        violations = check_all_invariants({"custom": 20.0}, invariants=custom)
        assert len(violations) == 1
        assert violations[0].invariant.name == "custom"


# ── should_trigger_mitigation ──────────────────────────────────────────


class TestShouldTriggerMitigation:
    def test_no_violations(self):
        assert should_trigger_mitigation([]) is False

    def test_warning_only(self):
        inv = PhysicsInvariant(name="b", description="", threshold=2.8, comparator="lt")
        v = PhysicsInvariantViolation(
            invariant=inv, actual_value=3.0, margin=0.2, severity="warning"
        )
        assert should_trigger_mitigation([v]) is False

    def test_critical_triggers(self):
        inv = PhysicsInvariant(name="b", description="", threshold=2.8, comparator="lt")
        v = PhysicsInvariantViolation(
            invariant=inv, actual_value=5.0, margin=2.2, severity="critical"
        )
        assert should_trigger_mitigation([v]) is True

    def test_mixed_critical_and_warning(self):
        inv1 = PhysicsInvariant(name="a", description="", threshold=1.0, comparator="gt")
        inv2 = PhysicsInvariant(name="b", description="", threshold=2.8, comparator="lt")
        v1 = PhysicsInvariantViolation(
            invariant=inv1, actual_value=0.9, margin=0.1, severity="warning"
        )
        v2 = PhysicsInvariantViolation(
            invariant=inv2, actual_value=5.0, margin=2.2, severity="critical"
        )
        assert should_trigger_mitigation([v1, v2]) is True
