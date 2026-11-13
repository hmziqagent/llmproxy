"""Python bindings for the llm-proxy engine.

The native extension (``llm_proxy._llm_proxy``) is a thin shim over the
shared bridge crate; this module unpacks its error tuple into a
:class:`ProxyError` with attributes and documents the surface:

>>> from llm_proxy import Proxy
>>> proxy = Proxy({"providers": [...]})
>>> reply = proxy.execute("acme", "anthropic_messages", request_body)
>>> with proxy.execute_stream("acme", "anthropic_messages", body) as stream:
...     for event in stream:
...         ...  # each event is a dict; the with block cancels an
...         ...  # unfinished stream
"""

from llm_proxy._llm_proxy import Proxy as _NativeProxy
from llm_proxy._llm_proxy import ProxyError as _NativeProxyError
from llm_proxy._llm_proxy import Stream as _NativeStream

__all__ = ["Proxy", "ProxyError", "Stream", "__version__"]

try:
    from llm_proxy._llm_proxy import __version__
except ImportError:  # pragma: no cover - built without the metadata
    __version__ = "0.0.0"


class ProxyError(Exception):
    """A request through the engine failed.

    Attributes:
        kind: the ledger's ``error_kind`` string (``UnknownProvider``, ...).
        status: the HTTP status the server would map this to.
        message: the sanitized client-facing message.
        retry_after_seconds: the advised wait, when the failure carries one.
        should_retry: whether the proxy wants the client to retry itself.
    """

    def __init__(self, kind, status, message, retry_after_seconds, should_retry):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        self.should_retry = should_retry


def _unpack(error):
    kind, status, message, retry_after_seconds, should_retry = error.args
    return ProxyError(kind, status, message, retry_after_seconds, should_retry)


class Stream:
    """A streaming response: an iterator of event dicts.

    Use ``with proxy.execute_stream(...) as stream:`` or call
    ``stream.cancel()`` (or ``stream.close()``); breaking out of a loop over
    a temporary stream cancels it when the object is released; keeping a
    reference keeps the request running. Leaving the ``with`` block (or
    releasing the stream) after the stream has run to its end does not
    cancel: the request settles as completed.
    """

    def __init__(self, native):
        self._native = native
        self._ended = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            event = self._native.__next__()
        except StopIteration:
            self._ended = True
            raise
        if "MessageStop" in event:
            self._ended = True
        return event

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # An ended stream must not be cancelled: the engine is settling it
        # as completed, and a cancel in that window would flip the row.
        if not self._ended:
            self.cancel()

    def cancel(self):
        """Stop delivering events and cancel the upstream request."""
        self._native.cancel()

    def close(self):
        """Alias of ``cancel()``."""
        self.cancel()

    def __del__(self):
        # A finalizer must not raise; this only makes the cancel prompt if
        # the native object outlives this wrapper.
        if self._ended:
            return
        try:
            self._native.cancel()
        except Exception:
            pass

    @property
    def cancelled(self):
        """Whether this stream has been cancelled."""
        return self._native.cancelled


class Proxy:
    """The llm-proxy engine as an in-process object.

    ``config`` is the bridge configuration dict: ``providers`` (same shape as
    the daemon's ``[[provider]]`` tables), ``request_timeout_secs`` and
    ``record_usage``.
    """

    def __init__(self, config):
        self._native = _NativeProxy(config)

    def execute(self, provider, dialect, body, options=None):
        """Run one non-streaming request end to end.

        Returns the response in the caller's dialect; raises ProxyError on
        failure.
        """
        try:
            return self._native.execute(provider, dialect, body, options)
        except _NativeProxyError as error:
            raise _unpack(error) from None

    def execute_stream(self, provider, dialect, body, options=None):
        """Run one streaming request; returns a :class:`Stream`."""
        try:
            return Stream(self._native.execute_stream(provider, dialect, body, options))
        except _NativeProxyError as error:
            raise _unpack(error) from None

    def translate_request(self, body, source, target, model=None):
        """Translate a request between wire dialects, with no I/O."""
        try:
            return self._native.translate_request(body, source, target, model)
        except _NativeProxyError as error:
            raise _unpack(error) from None

    def translate_response(self, body, source, target, model=None):
        """Translate a provider response into a client dialect, no I/O."""
        try:
            return self._native.translate_response(body, source, target, model)
        except _NativeProxyError as error:
            raise _unpack(error) from None

    def usage_events(self):
        """The recorded usage rows (dicts), when built with record_usage."""
        return self._native.usage_events()
