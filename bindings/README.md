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
Linux artifacts are built and install-tested here. Windows x64 artifacts are
produced by mingw-w64 cross-compilation and run-verified under wine 10.0
with the official Windows node.exe and embeddable CPython (the full Node
contract suite and the Python driver both pass; see the package READMEs);
the Node package also carries an msvc binary (cargo-xwin), verified the same
way. macOS artifacts are produced by zig cross-compilation (universal2 for
the Python wheel) and verified structurally. CI builds the Node binaries per
platform and runs the contract suite against them on the native legs
(`.github/workflows/node-binaries.yml`); every Python wheel is
install-tested by `.github/workflows/python-wheels.yml`.
