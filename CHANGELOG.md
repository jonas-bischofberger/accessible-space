# Changelog

## [2.1.0] - 2026-05-26

### Added
- `get_das_covered` function to compute defensive cover shadow
- `use_float32` parameter on public interface for increased computational efficiency
- `databallpy` as optional integration dependency + test cases

### Changed
- Core computation uses more efficient numpy methods, `scipy` dependency dropped.

### Fixed
- Minor edge cases (e.g. `infer_playing_direction` now skips empty periods like penalty shootouts)
