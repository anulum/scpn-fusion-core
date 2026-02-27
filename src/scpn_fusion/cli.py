# ──────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — Unified CLI
# © 1998–2026 Miroslav Šotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available
# ──────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click


LOGGER = logging.getLogger("scpn_fusion.cli")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODE_TIMEOUT_SECONDS = 1800.0


@dataclass(frozen=True)
class ModeSpec:
    module: str
    maturity: str
    description: str


MODE_SPECS: dict[str, ModeSpec] = {
    # Production/public modes.
    "kernel": ModeSpec("scpn_fusion.core.fusion_kernel", "public", "Grad-Shafranov equilibrium kernel"),
    "flight": ModeSpec("scpn_fusion.control.tokamak_flight_sim", "public", "Tokamak flight simulator"),
    "optimal": ModeSpec("scpn_fusion.control.fusion_optimal_control", "public", "Optimal control solver"),
    "learning": ModeSpec(
        "scpn_fusion.control.advanced_soc_fusion_learning",
        "public",
        "Advanced SOC fusion learning runtime",
    ),
    "digital-twin": ModeSpec("scpn_fusion.control.tokamak_digital_twin", "public", "Digital twin"),
    "control-room": ModeSpec("scpn_fusion.control.fusion_control_room", "public", "Control room simulator"),
    "sandpile": ModeSpec("scpn_fusion.core.sandpile_fusion_reactor", "public", "Legacy SOC research mode"),
    "nuclear": ModeSpec("scpn_fusion.nuclear.nuclear_wall_interaction", "public", "Wall interaction physics"),
    "breeding": ModeSpec("scpn_fusion.nuclear.blanket_neutronics", "public", "Blanket neutronics"),
    "safety": ModeSpec("scpn_fusion.control.disruption_predictor", "public", "Disruption predictor"),
    "optimizer": ModeSpec("scpn_fusion.core.compact_reactor_optimizer", "public", "Compact reactor optimizer"),
    "divertor": ModeSpec("scpn_fusion.core.divertor_thermal_sim", "public", "Divertor thermal model"),
    "diagnostics": ModeSpec("scpn_fusion.diagnostics.run_diagnostics", "public", "Diagnostics runtime"),
    "sawtooth": ModeSpec("scpn_fusion.core.mhd_sawtooth", "public", "MHD sawtooth model"),
    "geometry": ModeSpec("scpn_fusion.core.geometry_3d", "public", "3D geometry"),
    "spi": ModeSpec("scpn_fusion.control.spi_mitigation", "public", "SPI mitigation"),
    "scanner": ModeSpec("scpn_fusion.core.global_design_scanner", "public", "Global design scanner"),
    "heating": ModeSpec("scpn_fusion.core.rf_heating", "public", "RF heating"),
    "wdm": ModeSpec("scpn_fusion.core.wdm_engine", "public", "Warm dense matter engine"),
    "neuro-control": ModeSpec(
        "scpn_fusion.control.neuro_cybernetic_controller",
        "public",
        "Neuro-cybernetic controller",
    ),
    "rust-flight": ModeSpec(
        "scpn_fusion.control.rust_flight_sim_wrapper",
        "public",
        "Rust-native 10kHz flight simulator",
    ),
    # Surrogate modes (opt-in).
    "neural": ModeSpec("scpn_fusion.core.neural_equilibrium", "surrogate", "Neural equilibrium surrogate"),
    "fno-training": ModeSpec(
        "scpn_fusion.core.fno_jax_training",
        "surrogate",
        "JAX-accelerated FNO training",
    ),
    # Experimental modes (opt-in).
    "quantum": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum bridge"),
    "q-control": ModeSpec("scpn_fusion.core.quantum_bridge", "experimental", "Quantum control bridge"),
    "neuro-quantum": ModeSpec(
        "scpn_fusion.control.neuro_cybernetic_controller",
        "experimental",
        "Neuro-quantum control bridge",
    ),
    "lazarus": ModeSpec("scpn_fusion.core.lazarus_bridge", "experimental", "Lazarus bridge"),
    "director": ModeSpec("scpn_fusion.control.director_interface", "experimental", "Director interface"),
    "vibrana": ModeSpec("scpn_fusion.core.vibrana_bridge", "experimental", "Vibrana bridge"),
}


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _normalize_mode_timeout_seconds(timeout_s: float) -> float:
    timeout = float(timeout_s)
    if timeout <= 0.0:
        raise click.ClickException("--mode-timeout-seconds must be finite and > 0.")
    if timeout != timeout or timeout == float("inf") or timeout == float("-inf"):
        raise click.ClickException("--mode-timeout-seconds must be finite and > 0.")
    return timeout


def _available_modes(
    *,
    include_surrogate: bool,
    include_experimental: bool,
) -> list[str]:
    names: list[str] = []
    for mode, spec in MODE_SPECS.items():
        if spec.maturity == "public":
            names.append(mode)
        elif spec.maturity == "surrogate" and include_surrogate:
            names.append(mode)
        elif spec.maturity == "experimental" and include_experimental:
            names.append(mode)
    return names


def _lock_reason(
    mode: str,
    *,
    include_surrogate: bool,
    include_experimental: bool,
) -> str | None:
    spec = MODE_SPECS[mode]
    if spec.maturity == "surrogate" and not include_surrogate:
        return "surrogate mode locked; pass --surrogate or set SCPN_SURROGATE=1"
    if spec.maturity == "experimental" and not include_experimental:
        return "experimental mode locked; pass --experimental or set SCPN_EXPERIMENTAL=1"
    return None


def _execution_plan(
    mode: str,
    *,
    include_surrogate: bool,
    include_experimental: bool,
) -> list[str]:
    if mode == "all":
        return _available_modes(
            include_surrogate=include_surrogate,
            include_experimental=include_experimental,
        )

    if mode not in MODE_SPECS:
        all_modes = ", ".join(sorted(MODE_SPECS))
        raise click.ClickException(f"Unknown mode '{mode}'. Available modes: {all_modes}")

    reason = _lock_reason(
        mode,
        include_surrogate=include_surrogate,
        include_experimental=include_experimental,
    )
    if reason is not None:
        raise click.ClickException(f"Mode '{mode}' is locked: {reason}")
    return [mode]


def _run_mode(
    mode: str,
    *,
    python_bin: str,
    script_args: Sequence[str],
    mode_timeout_seconds: float,
    dry_run: bool,
) -> int:
    spec = MODE_SPECS[mode]
    cmd = [python_bin, "-m", spec.module, *script_args]
    LOGGER.info(
        "mode=%s maturity=%s module=%s cmd=%s",
        mode,
        spec.maturity,
        spec.module,
        shlex.join(cmd),
    )
    if dry_run:
        return 0
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=False,
            timeout=float(mode_timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        LOGGER.error(
            "Mode timed out: %s (timeout=%.1fs)",
            mode,
            float(mode_timeout_seconds),
        )
        return 124
    return int(result.returncode)


def _system_health_check() -> None:
    """Validate system resources and library versions."""
    LOGGER.info("Performing system health check...")
    
    # 1. Hardware Resources
    cpu_count = os.cpu_count() or 1
    if cpu_count < 2:
        LOGGER.warning("Low CPU core count (%d). Simulations may be slow.", cpu_count)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        avail_gb = mem.available / (1024**3)
        if total_gb < 8.0:
            LOGGER.warning("System RAM < 8GB (%.1f GB). Large grids may OOM.", total_gb)
        if avail_gb < 2.0:
            LOGGER.warning("Low available RAM (%.1f GB).", avail_gb)
    except ImportError:
        LOGGER.debug("psutil not installed; skipping memory check.")

    # 2. Library Versions
    try:
        import numpy as np
        if int(np.__version__.split('.')[0]) < 1:
            LOGGER.error("NumPy version too old: %s", np.__version__)
    except ImportError:
        LOGGER.critical("NumPy not found! Core physics will fail.")

    try:
        import scipy
        LOGGER.debug("SciPy version: %s", scipy.__version__)
    except ImportError:
        LOGGER.warning("SciPy not found. Some solvers will be unavailable.")

    # 3. Acceleration
    gpu_available = False
    try:
        import jax
        devices = jax.devices()
        gpu_available = any(d.platform == 'gpu' for d in devices)
    except ImportError:
        pass
    
    LOGGER.info(
        "Health check complete: CPUs=%d, GPU=%s",
        cpu_count, "YES" if gpu_available else "NO"
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("mode", required=False, default="all")
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
@click.option("--surrogate", is_flag=True, help="Unlock surrogate modes.")
@click.option("--experimental", is_flag=True, help="Unlock experimental modes.")
@click.option("--python-bin", default=sys.executable, show_default=True, help="Python executable for subprocesses.")
@click.option(
    "--mode-timeout-seconds",
    default=DEFAULT_MODE_TIMEOUT_SECONDS,
    show_default=True,
    type=float,
    help="Per-mode subprocess timeout in seconds.",
)
@click.option("--continue-on-error", is_flag=True, help="Continue running remaining modes after a failure.")
@click.option("--dry-run", is_flag=True, help="Print the launch plan without executing.")
@click.option("--list-modes", is_flag=True, help="List available modes and lock state, then exit.")
@click.option("--skip-health-check", is_flag=True, help="Skip startup system validation.")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="CLI log level.",
)
def cli(
    mode: str,
    script_args: tuple[str, ...],
    surrogate: bool,
    experimental: bool,
    python_bin: str,
    mode_timeout_seconds: float,
    continue_on_error: bool,
    dry_run: bool,
    list_modes: bool,
    skip_health_check: bool,
    log_level: str,
) -> None:
    """Unified SCPN Fusion launcher.

    MODE can be a specific simulation mode (e.g. ``kernel``) or ``all``
    to run all currently unlocked modes in sequence.
    """
    include_surrogate = surrogate or _env_enabled("SCPN_SURROGATE")
    include_experimental = experimental or _env_enabled("SCPN_EXPERIMENTAL")
    mode_timeout_seconds = _normalize_mode_timeout_seconds(mode_timeout_seconds)
    _configure_logging(log_level)

    if not skip_health_check and not dry_run and not list_modes:
        _system_health_check()

    if list_modes:
        click.echo("mode | maturity | unlocked | description")
        for name, spec in MODE_SPECS.items():
            unlocked = _lock_reason(
                name,
                include_surrogate=include_surrogate,
                include_experimental=include_experimental,
            ) is None
            click.echo(
                f"{name} | {spec.maturity} | {'yes' if unlocked else 'no'} | {spec.description}"
            )
        return

    if mode == "all" and script_args:
        raise click.ClickException("Positional script arguments are not supported with mode 'all'.")

    plan = _execution_plan(
        mode,
        include_surrogate=include_surrogate,
        include_experimental=include_experimental,
    )
    LOGGER.info("Resolved execution plan: %s", ", ".join(plan))

    failures: list[str] = []
    for planned_mode in plan:
        code = _run_mode(
            planned_mode,
            python_bin=python_bin,
            script_args=script_args,
            mode_timeout_seconds=mode_timeout_seconds,
            dry_run=dry_run,
        )
        if code != 0:
            failures.append(f"{planned_mode} (exit={code})")
            LOGGER.error("Mode failed: %s (exit=%d)", planned_mode, code)
            if not continue_on_error:
                break
        else:
            LOGGER.info("Mode succeeded: %s", planned_mode)

    if failures:
        raise click.ClickException("One or more modes failed: " + ", ".join(failures))

    LOGGER.info("Run completed successfully (%d mode(s)).", len(plan))


def main() -> int:
    try:
        cli.main(standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
