'use strict';

// The JS layer over the napi-generated loader (binding.js): it unpacks the
// native structured results into thrown ProxyErrors and an async-iterable
// stream. No engine logic lives here.

// The generated loader resolves this platform's prebuilt binary from the
// package directory or from the matching @freeoxide/llm-proxy-<platform>
// package (node-binaries.yml builds one per napi.targets entry).
const native = require('./binding.js');

class ProxyError extends Error {
  constructor(info) {
    super(info.message);
    this.name = 'ProxyError';
    this.kind = info.kind;
    this.status = info.status;
    this.message = info.message;
    this.retryAfterSeconds = info.retryAfterSeconds;
    this.shouldRetry = info.shouldRetry;
  }
}

function unwrap(result) {
  if (result.error) {
    throw new ProxyError(result.error);
  }
  return result.response;
}

// The native layer throws a napi error whose message is the JSON-encoded
// failure info (see src/lib.rs); keep it structured. A message that is not
// JSON (a napi type error, e.g. a config value that cannot be represented as
// JSON) is rethrown unchanged.
function toProxyError(error) {
  let info;
  try {
    info = JSON.parse(error.message);
  } catch {
    throw error;
  }
  return new ProxyError(info);
}

class Stream {
  constructor(nativeStream) {
    this._native = nativeStream;
  }

  async next() {
    return this._native.next();
  }

  async cancel() {
    return this._native.cancel();
  }

  get cancelled() {
    return this._native.cancelled;
  }

  [Symbol.asyncIterator]() {
    return {
      next: async () => {
        const event = await this._native.next();
        return event === null || event === undefined
          ? { done: true, value: undefined }
          : { done: false, value: event };
      },
      // for-await calls return() on an early exit only (break, return,
      // throw in the loop body) — never after the loop runs to completion —
      // so breaking out stops the upstream request like an explicit cancel.
      return: async () => {
        await this._native.cancel();
        return { done: true, value: undefined };
      },
      throw: async (error) => {
        await this._native.cancel();
        throw error;
      },
      [Symbol.asyncIterator]() {
        return this;
      },
    };
  }
}

// `await using` disposes the stream, stopping the upstream request. The
// symbol is missing on runtimes older than Node 20.4, so the method is
// attached with a computed key only when it exists.
if (Symbol.asyncDispose) {
  Object.defineProperty(Stream.prototype, Symbol.asyncDispose, {
    value: async function asyncDispose() {
      await this.cancel();
    },
    writable: true,
    configurable: true,
  });
}

class Proxy {
  constructor(config) {
    try {
      this._native = new native.Proxy(config);
    } catch (error) {
      throw toProxyError(error);
    }
  }

  async execute(provider, dialect, body, options) {
    return unwrap(await this._native.execute(provider, dialect, body, options));
  }

  async executeStream(provider, dialect, body, options) {
    let stream;
    try {
      stream = await this._native.executeStream(provider, dialect, body, options);
    } catch (error) {
      throw toProxyError(error);
    }
    return new Stream(stream);
  }

  translateRequest(body, source, target, model) {
    const result = this._native.translateRequest(body, source, target, model);
    if (result.error) {
      throw new ProxyError(result.error);
    }
    return result.result;
  }

  translateResponse(body, source, target, model) {
    const result = this._native.translateResponse(body, source, target, model);
    if (result.error) {
      throw new ProxyError(result.error);
    }
    return result.result;
  }

  usageEvents() {
    return this._native.usageEvents();
  }
}

module.exports = { Proxy, ProxyError, Stream };
