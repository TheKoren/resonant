# Contributing to resonant

Thank you for your interest in contributing to resonant! No contribution is too small.

## Getting Started

1. Fork and clone the repository
2. Run `cargo test --workspace` to ensure everything passes
3. Create a branch for your changes

## Development

### Prerequisites

- Rust stable (minimum 1.75)
- `cargo-llvm-cov` for coverage (optional, required for CI)
- `cargo-deny` for dependency auditing (optional, required for CI)

### Running checks locally

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
```

### Code quality targets

| Metric | Target |
|--------|--------|
| Test coverage | >= 80% per crate |
| Cyclomatic complexity | <= 10 per function |
| Nesting depth | <= 4 levels |
| Clippy warnings | Zero |

## Pull Requests

- Keep PRs focused — one feature or fix per PR
- Add tests for new behaviour
- Update documentation for public API changes
- Update the relevant crate's `CHANGELOG.md`
- Ensure all CI checks pass

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to build something useful.
