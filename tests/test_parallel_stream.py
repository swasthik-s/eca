from eca.training.parallel_stream import ParallelLaneByteStream


def test_parallel_lanes_keep_document_order_and_eos():
    docs = ["ab", "CD", "ef", "GH"]
    with ParallelLaneByteStream(docs, lanes=2, capacity_per_lane=16) as stream:
        rows = stream.read_batch(3)
        assert rows == [[97, 98, 256, 101], [67, 68, 256, 71]]
        rows2 = stream.read_batch(3)
        assert rows2 == [[102, 256, 256, 256], [72, 256, 256, 256]]
        assert stream.read_batch(3) == []


def test_parallel_stream_buffer_is_bounded():
    docs = ["x" * 1000 for _ in range(20)]
    with ParallelLaneByteStream(docs, lanes=4, capacity_per_lane=32,
                                low_watermark=0.25, high_watermark=0.75) as stream:
        # Let the producer run; the configured high watermark bounds every lane.
        import time
        time.sleep(0.05)
        assert stream.buffered_symbols <= 4 * 24
