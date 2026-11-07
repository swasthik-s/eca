"""Bounded producer/consumer streaming with low/high-watermark backpressure.

The producer reads source documents incrementally and appends raw UTF-8 bytes
plus EOS=256 to a bounded FIFO. The consumer trains on contiguous chunks.
No complete corpus is retained. The buffer is measured in byte-symbols.
"""
from __future__ import annotations

from collections import deque
from threading import Condition, Thread
from typing import Iterable


class BufferedByteStream:
    def __init__(
        self,
        documents: Iterable[str],
        capacity: int = 131072,
        low_watermark: float = 0.30,
        high_watermark: float = 0.90,
        eos_id: int = 256,
    ) -> None:
        if capacity < 2:
            raise ValueError("capacity must be at least 2")
        if not 0 < low_watermark < high_watermark <= 1:
            raise ValueError("Require 0 < low_watermark < high_watermark <= 1")
        self.capacity = capacity
        self.low = max(1, int(capacity * low_watermark))
        self.high = max(self.low + 1, int(capacity * high_watermark))
        self.high = min(capacity, self.high)
        self.eos_id = eos_id
        self.documents = iter(documents)
        self._queue: deque[int] = deque()
        self._cv = Condition()
        self._finished = False
        self._closed = False
        self._error: BaseException | None = None
        self._fetching = True
        self._thread = Thread(target=self._produce, name="eca-data-producer", daemon=True)
        self._thread.start()

    @property
    def buffered_symbols(self) -> int:
        with self._cv:
            return len(self._queue)

    def _produce(self) -> None:
        try:
            for document in self.documents:
                if not isinstance(document, str) or not document.strip():
                    continue
                # At most one source document is held at a time. Only a bounded
                # slice of its encoded bytes enters the queue.
                encoded = document.encode("utf-8", errors="replace")
                symbols = memoryview(encoded)
                index = 0
                while index <= len(symbols):
                    with self._cv:
                        while not self._closed and (
                            not self._fetching or len(self._queue) >= self.capacity
                        ):
                            self._cv.wait()
                        if self._closed:
                            return
                        room = min(self.capacity - len(self._queue),
                                   self.high - len(self._queue))
                        if room <= 0:
                            self._fetching = False
                            continue
                        take = min(room, len(symbols) - index)
                        if take:
                            self._queue.extend(symbols[index:index + take])
                            index += take
                        elif index == len(symbols):
                            self._queue.append(self.eos_id)
                            index += 1
                        if len(self._queue) >= self.high:
                            self._fetching = False
                        self._cv.notify_all()
        except BaseException as exc:
            with self._cv:
                self._error = exc
                self._cv.notify_all()
        finally:
            with self._cv:
                self._finished = True
                self._cv.notify_all()

    def read(self, count: int) -> list[int]:
        """Return up to count symbols, blocking until full or stream exhausted."""
        if count <= 0:
            raise ValueError("count must be positive")
        result: list[int] = []
        with self._cv:
            while len(result) < count:
                if self._queue:
                    take = min(count - len(result), len(self._queue))
                    for _ in range(take):
                        result.append(self._queue.popleft())
                    if len(self._queue) <= self.low:
                        self._fetching = True
                    self._cv.notify_all()
                elif self._error is not None:
                    raise RuntimeError("Data producer failed") from self._error
                elif self._finished:
                    break
                else:
                    self._cv.wait()
            if self._error is not None:
                raise RuntimeError("Data producer failed") from self._error
        return result

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=2)

    def __enter__(self) -> "BufferedByteStream":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
