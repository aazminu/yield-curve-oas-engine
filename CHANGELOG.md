# Changelog — `yield-curve-oas-engine`

All notable engineering changes to `yield-curve-oas-engine` (`yield_oas`) are documented in this file.

## [0.3.0] — 2026-10-24
### Added
- Documented CLI diagnostic usage (`python -m yield_oas --self-test --json`) and deterministic invariant verification workflows in `README.md`.
- Added release history and semantic versioning audit trail in `CHANGELOG.md`.

## [0.2.1] — 2026-10-23
### Added
- Added automated CLI and runtime invariant regression suite in `tests/test_diagnostics_cli.py` covering seed determinism, negative-seed rejection, JSON output schema, and `--self-test` exit codes.

## [0.2.0] — 2026-10-21
### Added
- Added `src/yield_oas/diagnostics.py` (`run_diagnostic_suite`, `validate_runtime_invariants`) and module CLI runner `src/yield_oas/__main__.py`.
- Exported diagnostic helpers in `yield_oas.__all__`.

## [0.1.0] — 2026-10-19
### Added
- Initial architecture and core test suite: Fixed-income par-to-zero yield curve bootstrapper, Nelson-Siegel-Svensson fitter, key-rate DV01 risk engine, and Option-Adjusted Spread (OAS) binomial lattice pricer.
