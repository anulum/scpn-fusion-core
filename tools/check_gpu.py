# ─────────────────────────────────────────────────────────────────────
# SCPN Fusion Core — GPU Availability Diagnostic
# © 1998–2026 Miroslav Sotek. All rights reserved.
# ─────────────────────────────────────────────────────────────────────
"""
Check GPU availability for JAX/PyTorch training.

Usage
-----
    python tools/check_gpu.py
    # Exit code 0 = GPU available, 1 = CPU only
"""

from __future__ import annotations

import sys


def main() -> None:
    print("=== SCPN Fusion Core — GPU Diagnostic ===\n")

    # ── JAX ───────────────────────────────────────────────────────
    jax_gpu = False
    try:
        import jax
        devices = jax.devices()
        jax_gpu = any(d.platform == "gpu" for d in devices)
        print(f"JAX version:   {jax.__version__}")
        print(f"JAX backend:   {'GPU' if jax_gpu else 'CPU'}")
        for d in devices:
            print(f"  Device:      {d}")
            if hasattr(d, "device_kind"):
                print(f"  Kind:        {d.device_kind}")
    except ImportError:
        print("JAX:           NOT INSTALLED")
    except Exception as e:
        print(f"JAX:           ERROR — {e}")

    # ── JAX CUDA info ────────────────────────────────────────────
    try:
        import jaxlib
        print(f"jaxlib version: {jaxlib.__version__}")
    except ImportError:
        pass

    # ── PyTorch ──────────────────────────────────────────────────
    torch_gpu = False
    try:
        import torch
        torch_gpu = torch.cuda.is_available()
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available:  {torch_gpu}")
        if torch_gpu:
            print(f"CUDA version:    {torch.version.cuda}")
            print(f"cuDNN version:   {torch.backends.cudnn.version()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    VRAM: {props.total_mem / 1e9:.1f} GB")
                print(f"    Compute: {props.major}.{props.minor}")
    except ImportError:
        print("\nPyTorch:         NOT INSTALLED")
    except Exception as e:
        print(f"\nPyTorch:         ERROR — {e}")

    # ── Rust wgpu ────────────────────────────────────────────────
    try:
        import scpn_fusion_rs
        if hasattr(scpn_fusion_rs, "py_gpu_available"):
            wgpu = scpn_fusion_rs.py_gpu_available()
            print(f"\nRust wgpu:       {'Available' if wgpu else 'Not available'}")
            if wgpu and hasattr(scpn_fusion_rs, "py_gpu_info"):
                info = scpn_fusion_rs.py_gpu_info()
                if info:
                    print(f"  Info:          {info}")
    except ImportError:
        pass

    # ── Summary ──────────────────────────────────────────────────
    gpu_ok = jax_gpu or torch_gpu
    print(f"\n{'='*50}")
    print(f"GPU Status: {'AVAILABLE' if gpu_ok else 'NOT AVAILABLE (CPU only)'}")
    if not gpu_ok:
        print("\nTo enable GPU training:")
        print("  pip install 'jax[cuda12]'")
        print("  # Requires NVIDIA GPU + CUDA 12.x toolkit")
    print(f"{'='*50}")

    sys.exit(0 if gpu_ok else 1)


if __name__ == "__main__":
    main()
