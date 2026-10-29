"""Latency simulation from measured encode/decode/compute timings."""
from __future__ import annotations

import numpy as np


def simulate_latency(*, edge_ms, encode_ms, transmitted_bits, bandwidth_bits_per_sec,
                     decode_ms, cloud_ms):
    if bandwidth_bits_per_sec <= 0 or transmitted_bits < 0:
        raise ValueError("Bandwidth must be positive and bit count nonnegative")
    return (edge_ms+encode_ms+1000.0*transmitted_bits/bandwidth_bits_per_sec+
            decode_ms+cloud_ms)


def latency_summary(samples_ms):
    a = np.asarray(samples_ms, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("Latency samples required")
    return {"p50_ms":float(np.percentile(a,50)), "p95_ms":float(np.percentile(a,95)),
            "mean_ms":float(a.mean()), "throughput_per_s":float(1000/a.mean()) if a.mean()>0 else None}
