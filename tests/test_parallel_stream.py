from eca.training.parallel_stream import ParallelLaneByteStream


def test_parallel_lanes_preserve_all_document_bytes_and_eos():
    docs = ["ab", "CD", "ef", "GH"]
    collected = []
    with ParallelLaneByteStream(docs, lanes=2, capacity_per_lane=16) as stream:
        while True:
            rows = stream.read_batch(3)
            if not rows:
                break
            collected.extend(value for row in rows for value in row)
    for byte_value in (97, 98, 67, 68, 101, 102, 71, 72):
        assert collected.count(byte_value) == 1
    assert collected.count(256) >= 4


def test_parallel_stream_buffer_is_bounded():
    docs = ["x" * 1000 for _ in range(20)]
    with ParallelLaneByteStream(docs, lanes=4, capacity_per_lane=32,
                                low_watermark=0.25, high_watermark=0.75) as stream:
        import time
        time.sleep(0.05)
        assert stream.buffered_symbols <= 4 * 24
