# @freeoxide/llm-proxy (Node)

Node bindings for the llm-proxy engine: translation between wire dialects,
request execution through the full pipeline, streaming events with
cancellation, and configuration from code. The native module wraps the
shared bridge crate (`crates/llm-proxy-bridge`) and adds no logic of its own.

The package name is scoped because the unscoped `llm-proxy` name on npm
belongs to an unrelated package.

## Install

The published package ships prebuilt binaries per platform, so no Rust
toolchain is needed at install time:

```
npm install @freeoxide/llm-proxy
```

The main package ships the loader (`binding.js`/`binding.d.ts`) and declares
one optional dependency per supported platform
(`@freeoxide/llm-proxy-<platform>`, listed in `package.json`); npm installs
only the one matching the running platform, and the loader requires from it
when no binary sits next to the package. Every platform binary is built in
CI by `.github/workflows/node-binaries.yml` (darwin x64/arm64, Linux
x64/arm64-gnu, Linux x64-musl, Windows x64/arm64-msvc); the native Linux,
Windows and macOS legs run the contract suite against the built binaries,
and a final clean-install job removes every Rust toolchain from the runner,
packs the main package and its platform package and installs both from
local tarballs with no compiler on PATH.

The package requires Node >= 18 (`engines` in `package.json`; the suite
runs on `node --test`, which needs 18).

## Use

```js
const { Proxy, ProxyError } = require('@freeoxide/llm-proxy');

const proxy = new Proxy({
  providers: [
    {
      name: 'acme',
      api_key: 'sk-...',
      auth_style: 'bearer',
      adapters: {
        messages: {
          protocol: 'anthropic_messages',
          endpoint: 'https://api.example.com/v1/messages',
        },
      },
      routes: { messages: 'messages' },
      model_aliases: { 'claude-sonnet-4-5': 'claude-sonnet-4-5-20250929' },
    },
  ],
  record_usage: true,
});

const reply = await proxy.execute('acme', 'anthropic_messages', requestBody);

const stream = await proxy.executeStream('acme', 'anthropic_messages', streamBody);
for await (const event of stream) {
  // event objects; break or cancel to stop upstream
}
await stream.cancel();

const translated = proxy.translateRequest(body, 'anthropic_messages', 'openai_chat_completions');
```

Dialect names: `anthropic_messages`, `openai_chat_completions`,
`openai_responses`, `gemini_generate_content`, `ollama_chat`,
`ollama_generate`. Provider protocols: `anthropic_messages`,
`openai_chat_completions`, `openai_responses`, `gemini_generate_content`.

Failures throw `ProxyError` with `kind`, `status`, `message`,
`retryAfterSeconds` and `shouldRetry`.

recorded usage rows are kept in memory for the life of the process; drain
with `usageEvents()` periodically, or leave `record_usage` off in
long-running services.

## Build and test

```
npm install
npm run build
npm test
```

The test suite replays the plan-02 contract fixtures from
`bindings/shared/cases.json` and compares against the snapshots the Rust
bridge test generates: the same fixtures must give the same results in Rust
and Node.
