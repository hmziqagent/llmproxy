/**
 * Node bindings for the llm-proxy engine (the JS layer over the napi
 * binding; see binding.d.ts for the raw native surface).
 */

/** Per-call options. */
export interface Options {
  /** The model named in the URL path (Gemini's generateContent). */
  model?: string;
  /** Inbound headers, subject to the provider's forward allowlist. */
  headers?: Record<string, string>;
}

/** A request through the engine failed. */
export declare class ProxyError extends Error {
  kind: string;
  status: number;
  message: string;
  retryAfterSeconds?: number | null;
  shouldRetry: boolean;
}

/** An event object from a streaming response (tagged core-event JSON). */
export type StreamEvent = Record<string, unknown>;

/**
 * A streaming response; async-iterable, cancellable. Leaving a
 * `for await` loop early (`break`, `return`, a throw in the body) stops
 * the upstream request through the iterator's `return()`, as does
 * `await using`; a loop that runs to completion does not cancel.
 */
export declare class Stream {
  next(): Promise<StreamEvent | null>;
  cancel(): Promise<void>;
  readonly cancelled: boolean;
  [Symbol.asyncIterator](): AsyncIterableIterator<StreamEvent>;
  /** Present only on runtimes that have `Symbol.asyncDispose` (Node 20.4+). */
  [Symbol.asyncDispose]?(): Promise<void>;
}

/**
 * Dialect names: anthropic_messages, openai_chat_completions,
 * openai_responses, gemini_generate_content, ollama_chat, ollama_generate.
 */
export type Dialect = string;

/** The engine wrapper: built from a configuration object. */
export declare class Proxy {
  constructor(config: Record<string, unknown>);
  execute(
    provider: string,
    dialect: Dialect,
    body: Record<string, unknown>,
    options?: Options
  ): Promise<Record<string, unknown>>;
  executeStream(
    provider: string,
    dialect: Dialect,
    body: Record<string, unknown>,
    options?: Options
  ): Promise<Stream>;
  translateRequest(
    body: Record<string, unknown>,
    source: string,
    target: string,
    model?: string
  ): Record<string, unknown>;
  translateResponse(
    body: Record<string, unknown>,
    source: string,
    target: string,
    model?: string
  ): Record<string, unknown>;
  usageEvents(): Array<Record<string, unknown>>;
}
