Scope the README control claims honestly (10 kHz kernel, Lean PID reference model)

Two README overclaims:
- "execute at 10 kHz+ against physics-informed plant models" (README control-first
  pitch) reads as a physics-in-loop real-time rate. The >10 kHz figure is the
  reduced-order control KERNEL on a simplified surrogate plant; a full free-boundary
  Grad-Shafranov solve in the loop runs at order 1 Hz. Restate it as the reduced-order
  kernel on a simplified plant and disclose that the physics-in-loop rate is far lower
  and is neither a same-work Rust-vs-Python speedup nor a physics-in-loop real-time rate.
- The "Verified properties" evidence cell described "normalized PID magnitudes remain
  bounded by actuator limits" without scoping it to the Lean reference model, implying
  the shipped controllers are verified. Consistent with the K17-A5 header correction,
  scope it explicitly to the PID saturation reference model (a Lean Nat contract, not an
  extraction of the shipped floating-point controllers, whose runtime bound is enforced
  in the actuator-boundary code).

verify_10khz_rust.py already flags mock mode honestly (timing_measured=False,
has_rust_backend, distinct mock success message), so no change there. sync_metadata
--check green; no test pins the old strings.

Addresses KIMI due-diligence finding K17-B3 ("10 kHz+ real-time" overclaim) and the
README propagation of K17-A5.

Seat: 14753

Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)
