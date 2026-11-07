"""Bounded multi-lane byte streaming for parallel ECA batches.

A background producer incrementally encodes documents and distributes whole
documents round-robin across independent lanes. Each lane has its own bounded
FIFO, so recurrent state can remain attached to the same lane across batches.
"""
from __future__ import annotations

from collections import deque
from threading import Condition, Thread
from typing import Iterable


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
        self._source = iter(documents)
        self._queues = [deque() for _ in range(lanes)]
        self._cv = Condition()
        self._finished = False
        self._closed = False
        self._error: BaseException | None = None
        self._thread = Thread(target=self._produce, name="eca-parallel-byte-producer", daemon=True)
        self._thread.start()

    def _append(self, lane: int, symbols: bytes | None) -> bool:
        values = memoryview(symbols) if symbols is not None else (self.eos_id,)
        index = 0
        while index < len(values):
            with self._cv:
                while not self._closed and len(self._queues[lane]) >= self.high:
                    self._cv.wait()
                if self._closed:
                    return False
                room = self.high - len(self._queues[lane])
                take = min(room, len(values) - index)
                self._queues[lane].extend(values[index:index + take])
                index += take
                self._cv.notify_all()
        return True

    def _produce(self) -> None:
        lane = 0
        try:
            for document in self._source:
                if not isinstance(document, str) or not document.strip():
                    continue
                encoded = document.encode("utf-8", errors="replace")
                if not encoded:
                    continue
                if not self._append(lane, encoded):
                    return
                if not self._append(lane, None):
                    return
                lane = (lane + 1) % self.lanes
        except BaseException as exc:
            with self._cv:
                self._error = exc
        finally:
            with self._cv:
                self._finished = True
                self._cv.notify_all()

    def read_batch(self, sequence_length: int) -> list[list[int]]:
        """Read one contiguous [lanes, sequence_length+1] batch."""
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        rows: list[list[int]] = []
        with self._cv:
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
                    elif self._error is not None:
                        raise RuntimeError("Parallel data producer failed") from self._error
                    elif self._finished:
                        if not row:
                            return rows
                        row.extend([self.eos_id] * (sequence_length + 1 - len(row)))
                    else:
                        self._cv.wait()
                rows.append(row)
        return rows

    @property
    def buffered_symbols(self) -> int:
        with self._cv:
            return sum(map(len, self._queues))

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=2)

    def __enter__(self) -> "ParallelLaneByteStream":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
