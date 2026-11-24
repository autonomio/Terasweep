"""Prepare single-channel return matrices from the retained market fixture."""

import argparse
import hashlib
import io
import json
import struct
import tarfile
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "benchmarks/archives/terasweep_reverse_funnel_experiment.tar.gz"
TRAIN_DATES = (date(2026, 1, 1), date(2026, 1, 23))
TEST_DATES = (date(2026, 1, 31), date(2026, 2, 6))


def prepare(output):
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        raw = archive.extractfile("terasweep-integrated/common/raw.parquet").read()
        provenance = json.load(
            archive.extractfile("terasweep-integrated/common/provenance.json")
        )
    if hashlib.sha256(raw).hexdigest() != provenance["sha256"]["raw.parquet"]:
        raise ValueError("Retained market fixture hash differs")
    market = pl.read_parquet(io.BytesIO(raw))
    output.mkdir(parents=True, exist_ok=True)
    for split, dates in (("train", TRAIN_DATES), ("test", TEST_DATES)):
        observations = market.filter(pl.col("datetime").dt.date().is_between(*dates))
        close = observations["close"].to_numpy()
        returns = ((close[1:] / close[:-1] - 1.0) * 100.0).astype("<f8")
        x, y = returns[:-1], returns[1:]
        (output / f"{split}.bin").write_bytes(
            struct.pack("<8sQQ", b"RIDGE001", len(x), 1) + x.tobytes() + y.tobytes()
        )
        if split == "test":
            opens = observations["open"].to_numpy()[1:-1].astype("<f8")
            closes = close[1:-1].astype("<f8")
            (output / "price.bin").write_bytes(
                struct.pack("<8sQ", b"PRICE001", len(x))
                + opens.tobytes()
                + closes.tobytes()
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
