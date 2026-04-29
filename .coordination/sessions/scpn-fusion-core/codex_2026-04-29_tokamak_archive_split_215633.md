---
agent: codex
session_start: 2026-04-29T21:56:33Z
repos_touched: [scpn-fusion-core]
tasks_completed: 1
incidents: 0
subagents_spawned: 0
subagents_spot_checked: 0
commits: 1
---

# Tokamak archive split

## Commit

- `ddd9a86` — `refactor: split tokamak archive shot wrappers`

## Work Completed

- Split disruption and synthetic NPZ shot wrapper APIs from `tokamak_archive.py` into `_tokamak_archive_shots.py`.
- Kept `tokamak_archive.py` as the public facade with stable exports through explicit imports and `__all__`.
- Updated the hardening linkage test to import `_tokamak_archive_shots.py` and assert facade re-export identity.
- Regenerated source backlog and underdeveloped register outputs. Source P0/P1 backlog now has 22 issue seeds, 0 P0, and 22 P1.

## Verification

- `python -m py_compile src/scpn_fusion/io/tokamak_archive.py src/scpn_fusion/io/_tokamak_archive_shots.py` — passed.
- `python -m ruff check src/scpn_fusion/io/tokamak_archive.py src/scpn_fusion/io/_tokamak_archive_shots.py tests/test_hardening_module_linkage.py` — passed; local config emitted the existing removed-rule warning for `UP038`.
- `python -m pytest tests/test_tokamak_archive.py tests/test_tokamak_archive_profiles.py tests/test_hardening_module_linkage.py -q` — 29 passed.
- `python tools/check_test_module_linkage.py` — passed with 0 unlinked modules.
- `python tools/generate_underdeveloped_register.py --check` — up to date with 120 entries.
- `python tools/generate_source_p0p1_issue_backlog.py --check` — up to date with 22 issue seeds.
- `git diff --cached --check` — passed before commit.
- Staged agent-name scan — clean before commit.
- Staged sensitive-string scan — clean before commit.
- `.coordination/FREEZE` and shared `FREEZE` — absent.

## Next Task

- Generated backlog has no P0 entries. Next generated item: P1 `src/scpn_fusion/control/burn_controller.py` with `SIMPLIFIED` marker.
