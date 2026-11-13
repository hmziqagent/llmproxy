# 10. Language bindings

- **Status:** Completed (2026-11-08)
- **Depends on:** 03, with a stable API
- **Audit rows:** None
- **Code-audit bugs:** None
- **Backlog:** None
- **Conflicts:** None

Give Python and Node users the in-process engine as native packages. These wrap
the engine and add no logic of their own, so they start once its API has stopped changing.

## To do

1. **Define the shared surface.** Cover translation, running a request, streaming events
   and configuration from code.
2. **Python package.** Check that `pyo3` and `maturin` are still maintained, then build
   wheels for Linux, macOS and Windows.
3. **Node package.** Check that `napi` is still maintained, then build prebuilt binaries
   for the same platforms.
4. **Share the tests.** Run the plan 02 contract fixtures through each binding.

## Done when

- `pip install` and `npm install` work on each platform with no Rust toolchain.
- The same fixtures give the same results through Rust, Python and Node.
- Streaming and cancellation work from each language.

## Done

- **Shared surface (step 1).** `crates/llm-proxy-bridge` is the JSON facade
  both bindings wrap: configuration from code (`BridgeConfig` — providers,
  aliases, routing, `request_timeout_secs`, `record_usage`), translation
  (`translate_request` / `translate_response`, pure codec work through the
  same client decoders, provider encoders and cross-format field policy the
  routes use), request execution (`execute` — the full engine pipeline with
  the caller's dialect decoded in and encoded back out, same-format
  passthrough included), and streaming events (`execute_stream` → an
  `EventStream` of tagged core-event JSON with `next`, `next_blocking` and
  `cancel`; dropping it before it ends, or calling `cancel`, settles the
  request as `ClientCancelled`, while a stream read to its end stays
  completed).
  Failures surface as `BridgeError` carrying `kind`, `status`, `message`,
  `retry_after` and `should_retry`. One process-wide tokio runtime is shared
  by every bridge (a per-bridge runtime panics when dropped from an async
  context).
- **Maintenance check (steps 2–3).** crates.io, 2026-11-07: `pyo3` 0.30.1
  (updated 2026-10-14, 298M downloads), `maturin` 1.16.0 (2026-10-29),
  `napi` 3.16.1 (2026-10-31), `napi-derive` 3.7.0, `napi-build` 2.6.0. The
  Node crate resolves napi 3.4.0/napi-derive 3.3.0 instead: the workspace
  MSRV (1.85) caps dependency resolution, and newer napi releases need a
  newer rustc. `@napi-rs/cli` ^3 comes from npm.
- **Python package (step 2).** `bindings/python`: pyo3 0.30 (abi3-py39) +
  maturin. The extension converts dicts/lists to JSON and back; the
  pure-Python layer (`llm_proxy/__init__.py`) unpacks the error tuple into a
  `ProxyError` with attributes — request failures and constructor config
  errors alike —, documents the surface, and gives `Stream` a
  context manager, `close()` and a finalizer: leaving a `with` block, an
  explicit `cancel()`/`close()`, or releasing the last reference cancels the
  upstream request and settles it as `ClientCancelled`, while releasing a
  stream that has run to its end leaves the completed settlement in place.
  Two wheels are
  built here: `llm_proxy-0.1.0-cp39-abi3-manylinux_2_34_x86_64.whl`
  (installed into a fresh venv whose `pip install` runs no Rust; 22 pytest
  tests pass against the installed wheel) and
  `llm_proxy-0.1.0-cp39-abi3-win_amd64.whl` (mingw-w64 cross-compilation),
  which is run-verified with the embeddable Windows CPython 3.13.5 under
  wine 10.0: every shared-contract case, the error shape, streaming,
  cancellation (including a watchdog `cancel()` releasing a reader thread
  blocked in `next()` on a stalled upstream) and the `ClientCancelled`
  settlement pass (`bindings/python/tests/windows_driver.py`). The macOS
  wheel is a
  universal2 build produced by zig cross-compilation (`maturin build
  --zig --target universal2-apple-darwin`; zig supplies the macOS libc
  stubs, `llvm-lipo` merges the slices): structurally verified as a fat
  `x86_64 + arm64` Mach-O whose undefined `Py_*` symbols resolve from the
  host interpreter and which links only system libraries. All the wheels
  plus an sdist are now built in CI by `.github/workflows/python-wheels.yml`
  (Linux x86_64 manylinux and musllinux_1_2, Linux aarch64 via the
  rust-cross manylinux container, Windows x64, macOS universal2), each wheel
  is checked to be `abi3`, and each OS wheel is pip-installed into a fresh
  venv (no Rust toolchain in the job) on ubuntu-22.04, windows-latest and
  macos-15 for CPython 3.9 and 3.13, with the pytest suite run against the
  install. The Node binaries are built per platform by
  `.github/workflows/node-binaries.yml`. aarch64 Windows is not built: it
  is optional and unverified. Both workflows also run a `clean-install`
  job on ubuntu-22.04, windows-latest and macos-15 that removes every Rust
  toolchain from the runner and fails if cargo/rustc/rustup stay
  resolvable, then does the consumer install — the wheel with
  `pip --no-index --only-binary=:all:` (an sdist build fails instead of
  compiling), the npm main package plus its platform package from local
  tarballs — and runs one no-network `translate_request`/`translateRequest`
  smoke against the installed package.
- **Node package (step 3).** `bindings/node`: napi-rs + `@napi-rs/cli`
  (`napi build --platform --release`). The native module returns structured
  results (`{response, error}` objects, JSON-encoded `ErrorInfo` on
  pre-commit stream failures and constructor config errors);
  `index.js` turns them into thrown
  `ProxyError`s and an async-iterable `Stream` whose `cancel()` settles as
  `ClientCancelled`, whose iterator `return()` cancels on a `for await`
  early exit (`break` or a throw in the loop body) and which defines
  `Symbol.asyncDispose` for `await using` on runtimes that have it. The
  `node --test` suite passes; the main npm package ships the loader
  (`binding.js`/`binding.d.ts`) plus one optional
  `@freeoxide/llm-proxy-<platform>` dependency per `napi.targets` entry
  (the `npm/<platform>/` scaffolding is committed), so `npm install` pulls
  the prebuilt binary for the running platform with no Rust toolchain. A
  Windows x64 binary (`llm-proxy.win32-x64-gnu.node`) is produced by
  mingw-w64 cross-compilation — the napi import library generated from the
  official `node.exe` exports (objdump export table → dlltool) — and
  run-verified: the full contract suite passed under wine 10.0 with the
  official node-v20.19.2 win-x64 `node.exe`. The npm name
  is scoped
  (`@freeoxide/llm-proxy`) because the unscoped `llm-proxy` on npm belongs
  to an unrelated package. An msvc binary
  (`llm-proxy.win32-x64-msvc.node`) is produced by cargo-xwin
  cross-compilation (the MSVC CRT/SDK libraries are fetched from Microsoft;
  on msvc the napi symbols resolve dynamically from the host node.exe at
  load, so no node import library is needed) and run-verified the same way:
  with only the msvc binary present, and with both Windows binaries
  present. Every platform binary (`llm-proxy.<platform>.node` for the seven
  `napi.targets` triples: darwin x64/arm64, Linux x64/arm64-gnu, Linux
  x64-musl, Windows x64/arm64-msvc) is built in CI by
  `.github/workflows/node-binaries.yml`; its native legs run the contract
  suite against the uploaded artifacts (linux-x64-gnu, win32-x64-msvc,
  darwin-arm64) and its cross-built legs are install-only. The darwin
  per-arch packages replace the old darwin-universal binary: the
  `bindings-macos` CI job and the msvc/gnu `require` fallback in `index.js`
  were removed with the native msvc CI build.
- **Shared tests (step 4).** `bindings/shared/cases.json` holds the shared
  cases: the plan-02 contract fixtures (Claude Code, the Anthropic SDK, the
  OpenAI SDK, Codex — captured and synthesized; provenance in the fixture
  README), the canned upstream responses extracted from the plan-02 contract
  replay harness (optional canned `status`/`headers`/`stall_ms`), the
  provider templates, the translation cases, the upstream error cases
  (429 with `Retry-After`, 502) and the cancellation scenario. The Rust bridge test
  (`crates/llm-proxy-bridge/tests/contract.rs`) replays every case and pins
  upstream request, upstream headers, client response, stream events and
  errors to snapshots under `bindings/shared/snapshots/` (regenerate with
  `UPDATE_SNAPSHOTS=1`; a shared normalization replaces wall-clock `created`
  stamps with a marker). The Python and Node suites replay the same cases
  against their own fake upstreams and compare to the same snapshots: the
  same fixtures give the same results through Rust, Python and Node
  (16 tests through Python and Node; the Rust bridge drives the same cases
  inside 4 aggregate tests). Streaming and cancellation are
  exercised in all three: events flow incrementally, `cancel()` stops
  delivery, the `ClientCancelled` usage row lands, and the proxy stays
  usable.
