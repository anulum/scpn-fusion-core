# Streamlit Demo Playbook (3 Minutes)

This playbook standardizes a public demo run for SCPN Fusion Core v3.5.0.

## Goal

Record a ~3 minute walkthrough showing:

1. Streamlit dashboard launch.
2. SPARC-like operating context.
3. Shot replay in the `Shot Replay` tab.
4. SNN control lane intervention context for disruption mitigation.

## One-Click Launch

```bash
docker compose up --build
```

Dashboard URL:

- `http://localhost:8501`

## Recommended Recording Flow

1. Start on landing screen with sidebar controls visible.
2. Open `Shot Replay` tab and select a disruption-labeled shot.
3. Show key metrics:
   - `Disruption Type`
   - `Is Disruption`
   - risk trajectory plot and threshold crossing
4. Narrate the SNN control path:
   - open controller mode in a second terminal:
     - `scpn-fusion neuro-control`
   - explain this runtime as the intervention lane used for mitigation benchmarks.
   - highlight inhibitor-arc safety interlocks as hard-stop logic for control commands.
5. End by showing reproducible startup command (`docker compose up --build`).

## Publishing Checklist

1. Export MP4 (1080p).
2. Upload to YouTube (public or unlisted).
3. Update `README.md` Public Demo section with the video URL.
4. Mirror link in release notes (`CHANGELOG.md`) for v3.5.0.
