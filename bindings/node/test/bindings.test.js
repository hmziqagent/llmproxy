'use strict';

// The shared binding contract, run through the Node package (plan 10, step
// 4): the plan-02 contract fixtures from bindings/shared/cases.json replay
// through llm-proxy against a local fake upstream, and the results must
// equal the checked-in snapshots the Rust bridge test generated — the same
// fixtures, the same results, in a third language. Streaming and
// cancellation behave exactly as in Rust.

const assert = require('node:assert/strict');
const { test } = require('node:test');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const { Proxy, ProxyError } = require('../index.js');

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const SHARED = path.join(REPO_ROOT, 'bindings', 'shared');
const shared = JSON.parse(fs.readFileSync(path.join(SHARED, 'cases.json'), 'utf8'));

function fixtureBody(relative) {
  return JSON.parse(fs.readFileSync(path.join(REPO_ROOT, shared.fixture_root, relative), 'utf8'));
}

// Serves one canned response (JSON or SSE, with gaps) and captures the last
// request, like the Rust harness's axum fake upstream.
class CannedUpstream {
  constructor(server, canned, gapMs) {
    this.server = server;
    this.canned = canned;
    this.gapMs = gapMs;
    this.capture = null;
  }

  requestHandler = (req, res) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => {
      let body;
      try {
        body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      } catch {
        body = {};
      }
      const headers = {};
      for (const [name, value] of Object.entries(req.headers)) {
        headers[name.toLowerCase()] = Array.isArray(value) ? value.join(', ') : value;
      }
      this.capture = { headers, body };
      if (this.canned.kind === 'json') {
        const payload = JSON.stringify(this.canned.body);
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(payload);
      } else {
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const events = this.canned.events;
        let index = 0;
        const sendNext = () => {
          if (index >= events.length) {
            res.end();
            return;
          }
          res.write(events[index]);
          index += 1;
          if (this.gapMs) {
            // unref'd so a multi-minute stall (a watchdog-cancel test) does
            // not keep the test process alive after the suite finishes.
            setTimeout(sendNext, this.gapMs).unref();
          } else {
            sendNext();
          }
        };
        sendNext();
      }
    });
  };

  get baseUrl() {
    const address = this.server.address();
    return `http://127.0.0.1:${address.port}`;
  }

  stop() {
    this.server.close();
  }
}

async function startUpstream(canned, gapMs = 0) {
  const upstream = new CannedUpstream(null, canned, gapMs);
  const server = http.createServer(upstream.requestHandler);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  upstream.server = server;
  return upstream;
}

function providerFor(name, baseUrl) {
  return JSON.parse(JSON.stringify(shared.providers[name]).replaceAll('{upstream}', baseUrl));
}

function headerSubset(headers) {
  const subset = {};
  for (const name of shared.header_subset) {
    if (name in headers) {
      subset[name] = headers[name];
    }
  }
  return subset;
}

// The shared normalization: JSON numbers under `created` are wall-clock
// stamps the chat encoder writes.
function normalize(value) {
  if (Array.isArray(value)) {
    return value.map(normalize);
  }
  if (value && typeof value === 'object') {
    const out = {};
    for (const [key, item] of Object.entries(value)) {
      out[key] =
        key === 'created' && typeof item === 'number' ? '<created-timestamp>' : normalize(item);
    }
    return out;
  }
  return value;
}

function errorInfo(failure) {
  return {
    kind: failure.kind,
    status: failure.status,
    message: failure.message,
    retry_after_secs: failure.retryAfterSeconds === undefined ? null : failure.retryAfterSeconds,
    should_retry: failure.shouldRetry,
  };
}

function snapshotFor(name) {
  return JSON.parse(fs.readFileSync(path.join(SHARED, 'snapshots', `${name}.json`), 'utf8'));
}

for (const testCase of shared.cases) {
  test(`case ${testCase.name} matches the shared snapshot`, async () => {
    const upstream = await startUpstream(shared.upstream_responses[testCase.upstream_response]);
    try {
      const provider = providerFor(testCase.provider, upstream.baseUrl);
      const proxy = new Proxy({ providers: [provider], record_usage: false });
      const body = fixtureBody(testCase.fixture);

      let response = null;
      let events = null;
      let error = null;
      if (testCase.stream) {
        try {
          const stream = await proxy.executeStream(
            testCase.provider,
            testCase.dialect,
            body,
            testCase.options
          );
          events = [];
          for await (const event of stream) {
            events.push(event);
          }
        } catch (failure) {
          if (!(failure instanceof ProxyError)) {
            throw failure;
          }
          error = errorInfo(failure);
        }
      } else {
        try {
          response = await proxy.execute(
            testCase.provider,
            testCase.dialect,
            body,
            testCase.options
          );
        } catch (failure) {
          if (!(failure instanceof ProxyError)) {
            throw failure;
          }
          error = errorInfo(failure);
        }
      }

      const captured = upstream.capture;
      const snapshot = {
        upstream_request: captured ? captured.body : null,
        upstream_headers: captured ? headerSubset(captured.headers) : null,
        response,
        stream_events: events,
        error,
      };
      assert.deepStrictEqual(normalize(snapshot), snapshotFor(testCase.name));
    } finally {
      upstream.stop();
    }
  });
}

for (const translate of shared.translates) {
  test(`translate ${translate.name} matches the shared snapshot`, () => {
    const config = {
      providers: [
        {
          name: 'offline',
          api_key: 'k',
          auth_style: 'bearer',
          adapters: {
            a: { protocol: 'anthropic_messages', endpoint: 'http://127.0.0.1:9/v1/messages' },
          },
          routes: { messages: 'a' },
        },
      ],
    };
    const proxy = new Proxy(config);
    const body = translate.fixture ? fixtureBody(translate.fixture) : translate.body;

    let result = null;
    let error = null;
    try {
      result =
        translate.mode === 'response'
          ? proxy.translateResponse(body, translate.from, translate.to)
          : proxy.translateRequest(body, translate.from, translate.to);
    } catch (failure) {
      if (!(failure instanceof ProxyError)) {
        throw failure;
      }
      error = errorInfo(failure);
    }
    assert.deepStrictEqual(normalize({ result, error }), snapshotFor(translate.name));
  });
}

test('failures throw ProxyError with status and kind', async () => {
  const upstream = await startUpstream(shared.upstream_responses.anthropic_json);
  try {
    const provider = providerFor('anthropic-upstream', upstream.baseUrl);
    const proxy = new Proxy({ providers: [provider] });
    const body = {
      model: 'claude-sonnet-4-5',
      max_tokens: 8,
      messages: [{ role: 'user', content: 'hi' }],
    };

    await assert.rejects(
      proxy.execute('nope', 'anthropic_messages', body),
      (failure) => {
        assert.ok(failure instanceof ProxyError);
        assert.equal(failure.kind, 'UnknownProvider');
        assert.equal(failure.status, 404);
        assert.equal(failure.shouldRetry, false);
        return true;
      }
    );
    await assert.rejects(
      proxy.execute('anthropic-upstream', 'anthropic_messages', { no: 'model' }),
      (failure) => {
        assert.ok(failure instanceof ProxyError);
        assert.equal(failure.status, 400);
        assert.equal(failure.kind, 'InvalidRequest');
        return true;
      }
    );
  } finally {
    upstream.stop();
  }
});

test('cancellation stops delivery and settles ClientCancelled', async () => {
  const scenario = shared.cancellation;
  const sse = await startUpstream(
    shared.upstream_responses[scenario.upstream_response],
    scenario.gap_ms || 250
  );
  const jsonUpstream = await startUpstream(shared.upstream_responses.anthropic_json);
  try {
    const proxy = new Proxy({
      record_usage: true,
      providers: [
        providerFor(scenario.stream_provider, sse.baseUrl),
        providerFor(scenario.followup_provider, jsonUpstream.baseUrl),
      ],
    });

    const stream = await proxy.executeStream(
      scenario.stream_provider,
      scenario.dialect,
      scenario.request
    );
    const iterator = stream[Symbol.asyncIterator]();
    const first = await iterator.next();
    assert.ok(!first.done);
    assert.ok('MessageStart' in first.value);

    await stream.cancel();
    assert.ok(stream.cancelled);
    const after = await stream.next();
    assert.equal(after, null);

    const deadline = Date.now() + (scenario.settle_timeout_secs || 5) * 1000;
    for (;;) {
      const rows = proxy.usageEvents();
      const cancelled = rows.some(
        (row) => row.kind === 'response_failed' && row.error_kind === 'ClientCancelled'
      );
      if (cancelled) {
        break;
      }
      if (Date.now() > deadline) {
        assert.fail(`the ClientCancelled row must land after cancel: ${JSON.stringify(rows)}`);
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    const followup = await proxy.execute(scenario.followup_provider, scenario.dialect, {
      model: 'claude-sonnet-4-5',
      max_tokens: 64,
      messages: [{ role: 'user', content: 'again' }],
    });
    assert.ok('id' in followup);
  } finally {
    sse.stop();
    jsonUpstream.stop();
  }
});

// The watchdog shape: while a next() call is parked on a stalled upstream
// (one SSE event, then nothing for an hour), cancel() must return promptly,
// unblock the pending next() with null and settle ClientCancelled — not
// wait out the 300 s request timeout.
test('cancel unblocks a pending next on a stalled upstream', async () => {
  const scenario = shared.cancellation;
  const sse = await startUpstream(
    shared.upstream_responses[scenario.upstream_response],
    3_600_000
  );
  const jsonUpstream = await startUpstream(shared.upstream_responses.anthropic_json);
  try {
    const proxy = new Proxy({
      record_usage: true,
      providers: [
        providerFor(scenario.stream_provider, sse.baseUrl),
        providerFor(scenario.followup_provider, jsonUpstream.baseUrl),
      ],
    });

    const stream = await proxy.executeStream(
      scenario.stream_provider,
      scenario.dialect,
      scenario.request
    );
    const iterator = stream[Symbol.asyncIterator]();
    const first = await iterator.next();
    assert.ok(!first.done);
    assert.ok('MessageStart' in first.value);

    // Park a next() on the stalled upstream, then cancel from a watchdog
    // timer, exactly as a consumer with a timeout would.
    const pending = stream.next();
    const cancelDeadline = Date.now() + 5_000;
    await new Promise((resolve) => setTimeout(resolve, 100));
    await stream.cancel();
    assert.ok(Date.now() < cancelDeadline, 'cancel() returns while next() is pending');
    assert.ok(stream.cancelled);

    const settled = await Promise.race([
      pending.then((event) => ({ event })),
      new Promise((resolve) => setTimeout(() => resolve({ timeout: true }), 5_000)),
    ]);
    assert.ok(!settled.timeout, 'the pending next() resolves after cancel');
    assert.equal(settled.event, null, 'the cancelled stream delivers no more events');

    const deadline = Date.now() + (scenario.settle_timeout_secs || 5) * 1000;
    for (;;) {
      const rows = proxy.usageEvents();
      const cancelled = rows.some(
        (row) => row.kind === 'response_failed' && row.error_kind === 'ClientCancelled'
      );
      if (cancelled) {
        break;
      }
      if (Date.now() > deadline) {
        assert.fail(
          `the ClientCancelled row must land after cancel: ${JSON.stringify(rows)}`
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }

    const followup = await proxy.execute(scenario.followup_provider, scenario.dialect, {
      model: 'claude-sonnet-4-5',
      max_tokens: 64,
      messages: [{ role: 'user', content: 'again' }],
    });
    assert.ok('id' in followup);
  } finally {
    sse.stop();
    jsonUpstream.stop();
  }
});

// Polls the usage ledger until the engine has settled a request as
// ClientCancelled; the cancel reaches the engine asynchronously.
async function waitForClientCancelledUsageRow(proxy, timeoutSecs, message) {
  const deadline = Date.now() + timeoutSecs * 1000;
  for (;;) {
    const rows = proxy.usageEvents();
    const cancelled = rows.some(
      (row) => row.kind === 'response_failed' && row.error_kind === 'ClientCancelled'
    );
    if (cancelled) {
      return;
    }
    if (Date.now() > deadline) {
      assert.fail(`${message}: ${JSON.stringify(rows)}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
}

// Breaking out of a for-await loop must stop the upstream request through
// the iterator's return(), exactly like an explicit cancel() would.
test('breaking out of for-await cancels the upstream request', async () => {
  const scenario = shared.cancellation;
  const sse = await startUpstream(
    shared.upstream_responses[scenario.upstream_response],
    3_600_000
  );
  try {
    const proxy = new Proxy({
      record_usage: true,
      providers: [providerFor(scenario.stream_provider, sse.baseUrl)],
    });
    const stream = await proxy.executeStream(
      scenario.stream_provider,
      scenario.dialect,
      scenario.request
    );

    for await (const event of stream) {
      assert.ok('MessageStart' in event);
      break;
    }

    assert.ok(stream.cancelled, 'breaking out of the loop cancels the stream');
    await waitForClientCancelledUsageRow(
      proxy,
      scenario.settle_timeout_secs || 5,
      'the ClientCancelled row must land after the break'
    );
  } finally {
    sse.stop();
  }
});

// A throw in the loop body is an early exit too: the runtime calls
// iterator.return() before propagating, which must cancel the stream.
test('an exception inside the loop body cancels as well', async () => {
  const scenario = shared.cancellation;
  const sse = await startUpstream(
    shared.upstream_responses[scenario.upstream_response],
    3_600_000
  );
  try {
    const proxy = new Proxy({
      record_usage: true,
      providers: [providerFor(scenario.stream_provider, sse.baseUrl)],
    });
    const stream = await proxy.executeStream(
      scenario.stream_provider,
      scenario.dialect,
      scenario.request
    );

    await assert.rejects(
      async () => {
        for await (const event of stream) {
          assert.ok('MessageStart' in event);
          throw new Error('consumer gave up');
        }
      },
      { message: 'consumer gave up' }
    );

    assert.ok(stream.cancelled, 'a throw in the loop body cancels the stream');
    await waitForClientCancelledUsageRow(
      proxy,
      scenario.settle_timeout_secs || 5,
      'the ClientCancelled row must land after the throw'
    );
  } finally {
    sse.stop();
  }
});

// return() fires only on early exit; a loop that runs to completion must
// leave the stream uncanceled and settle as completed.
test('completing the loop normally does not cancel', async () => {
  const scenario = shared.cancellation;
  const sse = await startUpstream(
    shared.upstream_responses[scenario.upstream_response],
    scenario.gap_ms || 250
  );
  try {
    const proxy = new Proxy({
      record_usage: true,
      providers: [providerFor(scenario.stream_provider, sse.baseUrl)],
    });
    const stream = await proxy.executeStream(
      scenario.stream_provider,
      scenario.dialect,
      scenario.request
    );

    let count = 0;
    let last = null;
    for await (const event of stream) {
      last = event;
      count += 1;
    }
    assert.ok(count >= 5, `the whole SSE response was consumed: ${count}`);
    assert.ok('MessageStop' in last, `the terminal event arrived: ${JSON.stringify(last)}`);

    assert.equal(stream.cancelled, false, 'normal completion does not cancel');
    const deadline = Date.now() + (scenario.settle_timeout_secs || 5) * 1000;
    for (;;) {
      if (proxy.usageEvents().some((row) => row.kind === 'response_completed')) {
        break;
      }
      if (Date.now() > deadline) {
        assert.fail(
          `the response_completed row must land: ${JSON.stringify(proxy.usageEvents())}`
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    const cancelled = proxy
      .usageEvents()
      .some((row) => row.kind === 'response_failed' && row.error_kind === 'ClientCancelled');
    assert.equal(cancelled, false, 'no ClientCancelled row after normal completion');
  } finally {
    sse.stop();
  }
});

test('usage events record the lifecycle when opted in', async () => {
  const upstream = await startUpstream(shared.upstream_responses.anthropic_json);
  try {
    const provider = providerFor('anthropic-upstream', upstream.baseUrl);
    const body = {
      model: 'claude-sonnet-4-5',
      max_tokens: 8,
      messages: [{ role: 'user', content: 'hi' }],
    };

    const recording = new Proxy({ providers: [provider], record_usage: true });
    await recording.execute('anthropic-upstream', 'anthropic_messages', body);
    const kinds = recording.usageEvents().map((row) => row.kind);
    assert.ok(kinds.includes('request_received'), `kinds: ${JSON.stringify(kinds)}`);
    assert.ok(kinds.includes('response_completed'), `kinds: ${JSON.stringify(kinds)}`);

    const silent = new Proxy({ providers: [provider], record_usage: false });
    await silent.execute('anthropic-upstream', 'anthropic_messages', body);
    assert.deepEqual(silent.usageEvents(), []);
  } finally {
    upstream.stop();
  }
});
