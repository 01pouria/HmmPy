# Changelog

## [0.3.3] - 2026-09-26

### Added
- Uniform `failed_checks` on typed decision results.
- `DecisionPolicy` with probability/confidence thresholds.
- Abstention and sample escalation.
- Optional decision history via `use_history`.
- Aggregated usage tracking across escalation.
- OpenAI-compatible provider abstraction.
- Token usage and optional cost estimation.

### Changed
- Project licensed under the MIT License.

### Notes
- Probability values are model-derived estimates and are not guaranteed to be calibrated.
