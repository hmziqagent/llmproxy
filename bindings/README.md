# Language bindings

Native packages wrapping the in-process engine through one shared surface,
`crates/llm-proxy-bridge` (JSON in, JSON out; no logic in the bindings):

- [`python/`](python/) — `llm-proxy` on PyPI (pyo3 + maturin, abi3 wheel)
- [`node/`](node/) — `@freeoxide/llm-proxy` on npm (napi-rs, prebuilt
  binaries; the unscoped `llm-proxy` name on npm belongs to an unrelated
  package)
- [`shared/`](shared/) — the binding contract: the plan-02 fixture cases,
  canned upstream responses and snapshots all three test suites share

Each package's suite replays the same plan-02 contract fixtures and compares
against the same snapshots, so the three implementations cannot drift apart.
Every artifact is built and verified in CI — nothing here depends on local
cross-compilation:

- `.github/workflows/python-wheels.yml` builds every Python wheel (Linux
  x86_64 manylinux and musllinux_1_2, Linux aarch64, Windows x64, macOS
  universal2) plus an sdist, and install-tests each OS wheel into a fresh
  venv — the full pytest suite runs against the installed wheel on Linux,
  Windows and macOS for CPython 3.9 and 3.13.
- `.github/workflows/node-binaries.yml` builds one prebuilt binary per
  `napi.targets` platform (darwin x64/arm64, Linux x64/arm64-gnu, Linux
  x64-musl, Windows x64/arm64-msvc) and runs the contract suite against the
  built binaries on the native Linux, Windows and macOS legs.

Both workflows end in a clean-install job that removes every Rust toolchain
from the runner, fails if cargo/rustc/rustup stay resolvable, and does the
consumer install; see the package READMEs and
`.github/workflows/bindings-release.yml` for the version-synced release
path.
