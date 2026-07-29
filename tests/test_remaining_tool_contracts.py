# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Offline behavioural contracts for the final top-level tool linkage tail."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATHS = {
    "check_gpu": "tools/check_gpu.py",
    "download_full_fidelity_public_sources": "tools/download_full_fidelity_public_sources.py",
    "download_qlknn10d": "tools/download_qlknn10d.py",
    "generate_benchmark_plots": "tools/generate_benchmark_plots.py",
    "qlknn10d_to_npz": "tools/qlknn10d_to_npz.py",
    "train_frc_surrogate": "tools/train_frc_surrogate.py",
    "train_neural_eq_v3": "tools/train_neural_eq_v3.py",
    "train_neural_equilibrium_augmented": "tools/train_neural_equilibrium_augmented.py",
    "train_neural_transport": "tools/train_neural_transport.py",
    "upgrade_notebook_to_golden_base": "tools/upgrade_notebook_to_golden_base.py",
}


def _load_tool(name: str) -> Any:
    module_name = f"remaining_tool_contracts_{name}"
    module_path = ROOT / TOOL_PATHS[name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_check_gpu_reports_mocked_optional_backends(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a mocked JAX accelerator without importing real GPU runtimes."""
    tool = _load_tool("check_gpu")
    fake_jax = SimpleNamespace(
        __version__="test-jax",
        devices=lambda: [SimpleNamespace(platform="gpu", device_kind="Mock Accelerator")],
    )
    fake_torch = SimpleNamespace(
        __version__="test-torch",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    fake_rust = SimpleNamespace(
        py_gpu_available=lambda: True,
        py_gpu_info=lambda: "mock-wgpu",
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "jaxlib", SimpleNamespace(__version__="test-jaxlib"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "scpn_fusion_rs", fake_rust)

    with pytest.raises(SystemExit) as exc_info:
        tool.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "JAX backend:   GPU" in output
    assert "Rust wgpu:       Available" in output
    assert "GPU Status: AVAILABLE" in output


class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> _BytesResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        payload, self._payload = self._payload, b""
        return payload


def test_public_source_downloader_builds_offline_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build git and web provenance through fully offline transport doubles."""
    tool = _load_tool("download_full_fidelity_public_sources")
    monkeypatch.setattr(tool, "ROOT", tmp_path)
    monkeypatch.setattr(tool, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(tool, "REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(tool, "JSON_REPORT", tmp_path / "reports" / "sources.json")
    monkeypatch.setattr(tool, "MD_REPORT", tmp_path / "reports" / "sources.md")
    monkeypatch.setattr(tool.shutil, "which", lambda _name: "/usr/bin/git")

    commands: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> None:
        commands.append(args)

    def fake_capture(args: list[str], **_kwargs: object) -> str:
        return "main" if "symbolic-ref" in args else "a" * 40

    monkeypatch.setattr(tool, "_run", fake_run)
    monkeypatch.setattr(tool, "_capture", fake_capture)
    monkeypatch.setattr(
        tool,
        "urlopen",
        lambda _request, timeout: _BytesResponse(f"payload-{timeout}".encode()),
    )
    git_source = tool.GitSource("repo", "https://example.invalid/repo.git", "repo", "gk", "test")
    web_source = tool.WebSource("page", "https://example.invalid", "page.html", "gk", "test")
    monkeypatch.setattr(tool, "GIT_SOURCES", (git_source,))
    monkeypatch.setattr(tool, "WEB_SOURCES", (web_source,))

    report = tool.build_report(timeout=7)
    tool.write_reports(report)

    assert report["all_reachable_downloads_completed"] is True
    assert [item["status"] for item in report["items"]] == ["downloaded", "downloaded"]
    assert commands[0][1:3] == ["clone", "--depth"]
    assert json.loads(tool.JSON_REPORT.read_text(encoding="utf-8"))["schema"] == (
        "full-fidelity-public-source-downloads.v1"
    )
    assert "| repo | gk | git | downloaded |" in tool.MD_REPORT.read_text(encoding="utf-8")


def test_qlknn_downloader_verifies_mocked_zenodo_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download and verify a mocked Zenodo file with checksum custody."""
    tool = _load_tool("download_qlknn10d")
    payload = b"qlknn-test-payload"
    digest = hashlib.md5(payload).hexdigest()
    metadata = {
        "files": [
            {
                "key": "sample.h5",
                "size": len(payload),
                "checksum": f"md5:{digest}",
                "links": {"self": "https://example.invalid/sample.h5"},
            }
        ]
    }
    monkeypatch.setattr(
        tool.urllib.request,
        "urlopen",
        lambda _request: _BytesResponse(json.dumps(metadata).encode()),
    )
    files = tool.fetch_zenodo_files()
    assert files[0]["filename"] == "sample.h5"
    assert tool._human_size(1024) == "1.0 KB"

    output_dir = tmp_path / "qlknn"

    def fake_download(_url: str, destination: Path, _expected_size: int = 0) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    monkeypatch.setattr(tool, "fetch_zenodo_files", lambda: files)
    monkeypatch.setattr(tool, "_download_with_progress", fake_download)
    tool.download(output_dir)

    assert tool._sha256(output_dir / "sample.h5") == hashlib.sha256(payload).hexdigest()
    assert tool.check(output_dir) is True
    assert tool.DOI in (output_dir / "README.md").read_text(encoding="utf-8")
    (output_dir / "sample.h5").write_bytes(b"corrupt")
    assert tool.check(output_dir) is False


def test_benchmark_plot_loader_filters_metrics_and_writes_png(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filter malformed campaign metrics and render a real PNG offline."""
    tool = _load_tool("generate_benchmark_plots")
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    report_path = tmp_path / "validation" / "reports" / "stress_test_campaign.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "controller": {
                    "p50_latency_us": 10,
                    "p95_latency_us": 20.5,
                    "p99_latency_us": 30,
                    "mean_reward": 1.5,
                },
                "bool-is-not-a-number": {
                    "p50_latency_us": True,
                    "p95_latency_us": 2,
                    "p99_latency_us": 3,
                },
            }
        ),
        encoding="utf-8",
    )

    data = tool._load_or_run_campaign(quick=True)
    assert data == {
        "controller": {
            "p50_latency_us": 10.0,
            "p95_latency_us": 20.5,
            "p99_latency_us": 30.0,
            "mean_reward": 1.5,
        }
    }
    output = tmp_path / "latency.png"
    tool.plot_controller_latency(data, output)
    assert output.read_bytes().startswith(b"\x89PNG")


def test_qlknn_converter_writes_deterministic_offline_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert a finite mocked HDF cohort into deterministic NPZ splits."""
    tool = _load_tool("qlknn10d_to_npz")
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    source = input_dir / "sample.h5"
    source.touch()
    output_dir = tmp_path / "processed"

    inputs = np.zeros((20, len(tool.QLKNN_INPUT_COLS)), dtype=np.float64)
    inputs[:, 0] = np.linspace(2.0, 7.0, 20)
    inputs[:, 1] = np.linspace(3.0, 8.0, 20)
    inputs[:, 3] = 2.0
    inputs[:, 4] = 0.5
    inputs[:, 5] = 0.4
    inputs[:, 6] = 1.0
    inputs[:, 7] = 1.0
    inputs[:, 8] = 0.01
    fluxes = np.column_stack(
        [
            np.linspace(-1.0, 2.0, 20),
            np.linspace(0.0, 3.0, 20),
            np.linspace(-2.0, 1.0, 20),
        ]
    )
    monkeypatch.setattr(tool, "_get_total_rows_hdf5", lambda _path: 20)
    monkeypatch.setattr(
        tool,
        "_load_chunk_hdf5",
        lambda _path, _start, _count: (inputs.copy(), fluxes.copy(), list(tool.QLKNN_INPUT_COLS)),
    )

    tool.process(input_dir, output_dir, max_samples=20, seed=9, gb_normalized=True)

    with np.load(output_dir / "train.npz", allow_pickle=False) as train:
        assert train["X"].shape == (18, 12)
        assert train["Y"].shape == (18, 3)
        assert np.all(train["Y"] >= 0.0)
        assert int(train["gb_normalized"]) == 1
    with np.load(output_dir / "val.npz", allow_pickle=False) as val:
        assert val["X"].shape == (1, 12)
    with np.load(output_dir / "test.npz", allow_pickle=False) as test:
        assert test["X"].shape == (1, 12)

    regimes = tool._classify_regime(
        np.array([2.0, 5.0, 2.0]),
        np.array([2.0, 2.0, 6.0]),
    )
    assert regimes.tolist() == [0, 1, 2]
    assert np.all(tool._gyrobohm_chi(np.array([1.0, 10.0])) > 0.0)


def test_frc_surrogate_small_jax_and_solver_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise compact FRC network math and finite mocked solver profiles."""
    tool = _load_tool("train_frc_surrogate")
    params = tool.init_mlp_params(jax.random.PRNGKey(1), 3, [4], 2)
    assert [tuple(layer["W"].shape) for layer in params] == [(3, 4), (4, 2)]
    prediction = tool.model_forward(params, jnp.ones(3))
    assert prediction.shape == (2,)
    loss = tool.mse_loss(params, jnp.ones((2, 3)), jnp.zeros((2, 2)))
    assert np.isfinite(float(loss))

    state = SimpleNamespace(
        converged=True,
        B_z=np.linspace(1.0, 2.0, 5, dtype=np.float64),
    )
    monkeypatch.setattr(tool, "solve_frc_equilibrium", lambda *_args, **_kwargs: state)
    features, profiles = tool.generate_frc_data(2, 5, seed=3)
    assert features.shape == (2, 7)
    assert profiles.shape == (2, 5)
    assert np.all(features[:, 3] == 0.0)


def test_neural_eq_v3_pca_mlp_and_stratified_split() -> None:
    """Preserve PCA reconstruction, MLP shape, and stratified split contracts."""
    tool = _load_tool("train_neural_eq_v3")
    samples = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]],
        dtype=np.float64,
    )
    pca = tool.MinimalPCA(k=2)
    latent = pca.fit_transform(samples)
    reconstructed = pca.inverse_transform(latent)
    np.testing.assert_allclose(reconstructed, samples, atol=1e-12)
    assert pca.evr is not None and float(np.sum(pca.evr)) == pytest.approx(1.0)

    mlp = tool.MLP([3, 4, 2], seed=5)
    assert mlp.forward(samples[:2]).shape == (2, 2)
    labels = ["a"] * 10 + ["b"] * 10
    train, val, test = tool.stratified_split(labels, np.random.default_rng(7))
    assert (len(train), len(val), len(test)) == (14, 2, 4)
    assert len(set(train) | set(val) | set(test)) == 20


def test_augmented_equilibrium_file_collection_and_empty_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Collect machine files deterministically and fail closed on no data."""
    tool = _load_tool("train_neural_equilibrium_augmented")
    ref_dir = tmp_path / "validation" / "reference_data"
    (ref_dir / "machine_b").mkdir(parents=True)
    (ref_dir / "machine_a").mkdir()
    (ref_dir / "machine_b" / "b.eqdsk").touch()
    (ref_dir / "machine_a" / "a.geqdsk").touch()
    assert [path.name for path in tool.collect_geqdsk_files(ref_dir)] == ["a.geqdsk", "b.eqdsk"]

    empty_root = tmp_path / "empty"
    (empty_root / "validation" / "reference_data").mkdir(parents=True)
    monkeypatch.setattr(tool, "REPO_ROOT", empty_root)
    monkeypatch.setattr(sys, "argv", ["train_neural_equilibrium_augmented.py"])
    assert tool.main() == 1
    assert "No GEQDSK/EQDSK files found" in capsys.readouterr().out


def test_neural_transport_small_training_math_is_finite() -> None:
    """Keep the compact neural-transport generation and update path finite."""
    tool = _load_tool("train_neural_transport")
    features, targets = tool.generate_synthetic_data(8)
    assert features.shape == (8, tool.INPUT_DIM)
    assert targets.shape == (8, tool.OUTPUT_DIM)
    assert bool(jnp.all(targets >= 0.0))

    params = tool.init_params(jax.random.PRNGKey(4))
    before = float(tool.loss_fn(params, features, targets))
    updated = tool.update(params, features, targets, lr=1e-5)
    after = float(tool.loss_fn(updated, features, targets))
    assert np.isfinite(before)
    assert np.isfinite(after)
    assert tool.forward(updated, features[0]).shape == (tool.OUTPUT_DIM,)


def test_notebook_upgrader_rewrites_minimal_notebook_without_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rewrite the notebook protocol deterministically without executing cells."""

    class Cell(dict[str, Any]):
        @property
        def source(self) -> str:
            return str(self["source"])

        @source.setter
        def source(self, value: str) -> None:
            self["source"] = value

    cells = [Cell(source="", id=f"cell-{index}") for index in range(22)]
    cells[20].source = 'artifact = DeploymentArtifact(name="hero_neuro_symbolic_control")'
    notebook = SimpleNamespace(cells=cells)
    written: list[tuple[Any, Path]] = []

    def fake_read(_source: Path, *, as_version: int) -> Any:
        assert as_version == 4
        return notebook

    def fake_write(value: Any, destination: Path) -> None:
        written.append((value, destination))
        destination.write_text("offline notebook protocol stub\n", encoding="utf-8")

    fake_nbformat = ModuleType("nbformat")
    fake_nbformat.__dict__.update(read=fake_read, write=fake_write)
    monkeypatch.setitem(sys.modules, "nbformat", fake_nbformat)
    tool = _load_tool("upgrade_notebook_to_golden_base")
    monkeypatch.chdir(tmp_path)
    examples = tmp_path / "examples"
    examples.mkdir()
    source = examples / "neuro_symbolic_control_demo.ipynb"
    source.write_text("offline source notebook\n", encoding="utf-8")

    assert tool._replace_once("a-a", "a", "b") == "b-a"
    tool.main()

    output = examples / "neuro_symbolic_control_demo_v2.ipynb"
    assert written[0][0] is notebook
    assert written[0][1].resolve() == output
    upgraded = written[0][0]
    assert upgraded.cells[0].source.startswith("# Neuro-Symbolic Control Demo (Golden Base v2)")
    assert "golden_base_neuro_symbolic_control" in upgraded.cells[20].source
    assert all("id" not in cell for cell in upgraded.cells)
