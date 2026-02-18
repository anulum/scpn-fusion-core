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
    # Surrogate modes (opt-in).
    "neural": ModeSpec("scpn_fusion.core.neural_equilibrium", "surrogate", "Neural equilibrium surrogate"),
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
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(result.returncode)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("mode", required=False, default="all")
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
@click.option("--surrogate", is_flag=True, help="Unlock surrogate modes.")
@click.option("--experimental", is_flag=True, help="Unlock experimental modes.")
@click.option("--python-bin", default=sys.executable, show_default=True, help="Python executable for subprocesses.")
@click.option("--continue-on-error", is_flag=True, help="Continue running remaining modes after a failure.")
@click.option("--dry-run", is_flag=True, help="Print the launch plan without executing.")
@click.option("--list-modes", is_flag=True, help="List available modes and lock state, then exit.")
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
    continue_on_error: bool,
    dry_run: bool,
    list_modes: bool,
    log_level: str,
) -> None:
    """Unified SCPN Fusion launcher.

    MODE can be a specific simulation mode (e.g. ``kernel``) or ``all``
    to run all currently unlocked modes in sequence.
    """
    include_surrogate = surrogate or _env_enabled("SCPN_SURROGATE")
    include_experimental = experimental or _env_enabled("SCPN_EXPERIMENTAL")
    _configure_logging(log_level)

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

