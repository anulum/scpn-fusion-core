# Streamlit Demo Playbook (3 Minutes)

## Audience and use case

This playbook is a reproducible script for external demo preparation. It is
designed for stakeholder briefings, investor-facing walkthroughs, and reviewable
internal presentations where claims must remain linked to commands and artifacts.

A successful demo flow must be accompanied by:

- command evidence (launch and interaction steps)
- artifact logs or screenshots for the exact run
- a clear statement of whether results are exploratory or accepted evidence

## Intended role

This document is the public-facing standard runbook for demo production. It
separates reproducible demonstration preparation from validation acceptance and
keeps every demo run aligned with the linked artifact and benchmark chain.

## Demo scope

This playbook defines the standard public demonstration flow and required sequence for reproducible and reviewable Streamlit demos.

## What this page covers

This document defines the reviewable demo sequence for public walkthroughs.
Use it when generating externally visible media so that claims remain tied to
reproducible launch commands and active benchmark artifacts.

This playbook standardizes a public demo run for SCPN Fusion Core v3.10.0.

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

## Publishing Criteria

1. Export MP4 (1080p).
2. Upload to YouTube (public or unlisted).
3. Update `README.md` Public Demo section with the video URL.
4. Mirror link in release notes (`CHANGELOG.md`) for v3.5.0.

## Review criteria

A recorded demo is only used as evidence when its runtime command, screenshot
set, and report linkage are preserved together. If a recording only shows
interface behavior without commands and benchmark references, keep it labeled
as internal demonstration material.
