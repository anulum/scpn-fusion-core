# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Fusion Core — Surrogate UQ Cards Tests
"""Tests for the per-surrogate UQ-card generator and guard (master-plan T-4)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "generate_surrogate_uq_cards.py"
MANIFEST = ROOT / "validation" / "surrogate_uq_cards.json"
OUTPUT = ROOT / "docs" / "SURROGATE_UQ_CARDS.md"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_surrogate_uq_cards", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_manifest(tmp_path: Path, payload: dict[str, Any]) -> Path:
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


class TestCommittedManifest:
    """The committed manifest is fully anchored and the page is fresh."""

    def test_committed_manifest_verifies_clean(self) -> None:
        module = _load_module()
        _, cards = module.load_manifest(MANIFEST)
        assert module.verify_cards(cards, ROOT) == []

    def test_committed_page_is_up_to_date(self) -> None:
        module = _load_module()
        assert module.main(["--check"]) == 0

    def test_covers_expected_surrogate_lanes(self) -> None:
        module = _load_module()
        _, cards = module.load_manifest(MANIFEST)
        ids = {card.card_id for card in cards}
        assert {
            "qlknn10d_neural_transport",
            "disruption_risk_predictor",
            "itpa_pretrained_mlp",
            "fno_turbulence_suppressor",
            "tglf_surrogate_bridge",
            "iter_gpu_surrogate",
        } <= ids

    def test_deprecated_fno_stays_scoped(self) -> None:
        """Acceptance: the deprecated FNO lane stays scoped."""
        module = _load_module()
        _, cards = module.load_manifest(MANIFEST)
        fno = next(card for card in cards if card.card_id == "fno_turbulence_suppressor")
        assert fno.status == "deprecated_scoped"
        assert any("deprecated_default_lane_guard" in anchor.file for anchor in fno.ood_anchors)

    def test_every_card_names_ood_and_fallback(self) -> None:
        module = _load_module()
        _, cards = module.load_manifest(MANIFEST)
        for card in cards:
            assert card.ood_mechanism
            assert card.fallback.summary

    def test_rendered_page_contains_all_cards(self) -> None:
        module = _load_module()
        scope_note, cards = module.load_manifest(MANIFEST)
        rendered = module.render_markdown(scope_note, cards, "validation/surrogate_uq_cards.json")
        for card in cards:
            assert f"### `{card.card_id}`" in rendered


class TestManifestValidation:
    """Structural validation fails closed on malformed manifests."""

    def test_rejects_foreign_schema(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["schema"] = "other.schema"
        with pytest.raises(ValueError, match="unexpected manifest schema"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_duplicate_card_id(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["cards"].append(dict(payload["cards"][0]))
        with pytest.raises(ValueError, match="Duplicate card id"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_unknown_status(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["cards"][0]["status"] = "experimental"
        with pytest.raises(ValueError, match="status must be one of"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_empty_model_artifacts(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["cards"][0]["model_artifacts"] = []
        with pytest.raises(ValueError, match="model_artifacts must not be empty"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_non_numeric_threshold(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["cards"][0]["ood"]["thresholds"] = {"soft_sigma": "three"}
        with pytest.raises(ValueError, match="must be a number"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_missing_ood_section(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        del payload["cards"][0]["ood"]
        with pytest.raises(ValueError, match="ood must be an object"):
            module.load_manifest(_write_manifest(tmp_path, payload))

    def test_rejects_empty_cards(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["cards"] = []
        with pytest.raises(ValueError, match="cards must be a non-empty list"):
            module.load_manifest(_write_manifest(tmp_path, payload))


class TestAnchorVerification:
    """Anchor verification catches removed or renamed linked entities."""

    def _verify_with_mutation(self, mutate: Any) -> list[str]:
        module = _load_module()
        payload = _manifest_payload()
        mutate(payload)
        target = ROOT / "validation" / "_tmp_uq_cards_test.json"
        try:
            target.write_text(json.dumps(payload), encoding="utf-8")
            _, cards = module.load_manifest(target)
            return list(module.verify_cards(cards, ROOT))
        finally:
            target.unlink(missing_ok=True)

    def test_detects_missing_model_file(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["cards"][0]["model_artifacts"] = ["src/scpn_fusion/core/no_such_model.py"]

        errors = self._verify_with_mutation(mutate)
        assert any("file not found" in error for error in errors)

    def test_detects_missing_symbol(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["cards"][0]["model_artifacts"] = [
                "src/scpn_fusion/core/_neural_transport_runtime.py::NoSuchModel"
            ]

        errors = self._verify_with_mutation(mutate)
        assert any("symbol not found" in error for error in errors)

    def test_rejects_symbol_anchor_on_non_python_file(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            payload["cards"][0]["calibration"]["anchors"] = [
                "validation/reports/transport_uncertainty_envelope_benchmark.json::field"
            ]

        errors = self._verify_with_mutation(mutate)
        assert any("symbol anchors unsupported" in error for error in errors)

    def test_non_promoted_card_without_gaps_flagged(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            for card in payload["cards"]:
                if card["id"] == "tglf_surrogate_bridge":
                    card["gaps"] = []

        errors = self._verify_with_mutation(mutate)
        assert any("must declare its gaps" in error for error in errors)


class TestCliModes:
    """CLI write and check modes behave fail-closed."""

    def test_write_mode_creates_output(self, tmp_path: Path) -> None:
        module = _load_module()
        output = tmp_path / "cards.md"
        assert module.main(["--output", str(output)]) == 0
        assert "# Surrogate UQ Cards" in output.read_text(encoding="utf-8")

    def test_check_mode_fails_on_missing_output(self, tmp_path: Path) -> None:
        module = _load_module()
        assert module.main(["--output", str(tmp_path / "missing.md"), "--check"]) == 1

    def test_check_mode_fails_on_stale_output(self, tmp_path: Path) -> None:
        module = _load_module()
        output = tmp_path / "cards.md"
        assert module.main(["--output", str(output)]) == 0
        output.write_text(output.read_text(encoding="utf-8") + "\nedit\n", encoding="utf-8")
        assert module.main(["--output", str(output), "--check"]) == 1

    def test_broken_anchor_fails_write_mode(self, tmp_path: Path) -> None:
        module = _load_module()
        payload = _manifest_payload()
        payload["cards"][0]["model_artifacts"] = ["src/missing.py"]
        manifest = _write_manifest(tmp_path, payload)
        output = tmp_path / "cards.md"
        assert module.main(["--manifest", str(manifest), "--output", str(output)]) == 1
        assert not output.exists()
