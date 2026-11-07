"""Bounded multi-lane byte streaming with one producer per GPU lane.

Workers fetch documents from a shared lazy iterator under a short lock, then
encode and fill their own bounded FIFO independently. This prevents a full
lane from blocking production for every other lane.
"""
from __future__ import annotations

from collections import deque
from threading import Condition, Lock, Thread
from typing import Iterable, Iterator


class ParallelLaneByteStream:
    def __init__(
        self,
        documents: Iterable[str],
        lanes: int,
        capacity_per_lane: int = 65536,
        low_watermark: float = 0.30,
        high_watermark: float = 0.90,
        eos_id: int = 256,
    ) -> None:
        if lanes < 1 or capacity_per_lane < 2:
            raise ValueError("lanes must be >= 1 and capacity_per_lane >= 2")
        if not 0 < low_watermark < high_watermark <= 1:
            raise ValueError("Require 0 < low_watermark < high_watermark <= 1")
        self.lanes = lanes
        self.capacity = capacity_per_lane
        self.low = max(1, int(capacity_per_lane * low_watermark))
        self.high = min(capacity_per_lane, max(self.low + 1, int(capacity_per_lane * high_watermark)))
        self.eos_id = eos_id
        self._source: Iterator[str] = iter(documents)
        self._source_lock = Lock()
        self._queues = [deque() for _ in range(lanes)]
        self._cv = Condition()
        self._done = [False] * lanes
        self._closed = False
        self._errors: list[BaseException | None] = [None] * lanes
        self._threads = [
            Thread(target=self._produce_lane, args=(lane,), name=f"eca-byte-producer-{lane}", daemon=True)
            for lane in range(lanes)
        ]
        for thread in self._threads:
            thread.start()

    def _next_document(self) -> str | None:
        with self._source_lock:
            try:
                return next(self._source)
            except StopIteration:
                return None

    def _append(self, lane: int, values: bytes | tuple[int, ...]) -> bool:
        index = 0
        while index < len(values):
            with self._cv:
                while not self._closed and len(self._queues[lane]) >= self.high:
                    self._cv.wait()
                if self._closed:
                    return False
                room = min(self.high, self.capacity) - len(self._queues[lane])
                take = min(room, len(values) - index)
                if take <= 0:
                    continue
                self._queues[lane].extend(values[index:index + take])
                index += take
                self._cv.notify_all()
        return True

    def _produce_lane(self, lane: int) -> None:
        try:
            while not self._closed:
                document = self._next_document()
                if document is None:
                    break
                if not isinstance(document, str) or not document.strip():
                    continue
                encoded = document.encode("utf-8", errors="replace")
                if encoded and not self._append(lane, encoded):
                    return
                if encoded and not self._append(lane, (self.eos_id,)):
                    return
        except BaseException as exc:
            self._errors[lane] = exc
        finally:
            with self._cv:
                self._done[lane] = True
                self._cv.notify_all()

    def read_batch(self, sequence_length: int) -> list[list[int]]:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        rows: list[list[int]] = []
        with self._cv:
            if all(done and not queue for done, queue in zip(self._done, self._queues)):
                errors = [e for e in self._errors if e is not None]
                if errors:
                    raise RuntimeError("Parallel data producer failed") from errors[0]
                return []
            for lane in range(self.lanes):
                row: list[int] = []
                while len(row) < sequence_length + 1:
                    queue = self._queues[lane]
                    if queue:
                        take = min(sequence_length + 1 - len(row), len(queue))
                        for _ in range(take):
                            row.append(queue.popleft())
                        if len(queue) <= self.low:
                            self._cv.notify_all()
                    elif self._errors[lane] is not None:
                        raise RuntimeError(f"Data producer for lane {lane} failed") from self._errors[lane]
                    elif self._done[lane]:
                        if not row:
                            break
                        row.extend([self.eos_id] * (sequence_length + 1 - len(row)))
                    else:
                        self._cv.wait()
                if row:
                    rows.append(row)
        return rows

    @property
    def buffered_symbols(self) -> int:
        with self._cv:
            return sum(len(queue) for queue in self._queues)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        for thread in self._threads:
            thread.join(timeout=2)

    def __enter__(self) -> "ParallelLaneByteStream":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
