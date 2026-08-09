"""Turn a benchmark .npz into a tidy Parquet table plus a metadata JSON.

Keys in the archive look like ``n500__muladd__quad-sleef__out__cold`` and map to
a 1-D array of per-repeat timings in microseconds. Runs that sweep a single size
omit the leading ``n`` field. Combinations that were never run (too slow, out of
memory budget) are simply absent; they are emitted as null timings so the table
is a complete grid.
"""

import itertools
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

DATA_ROOT = Path("data")
FIELDS = ("n", "operation", "implementation", "destination", "cache_state")
FIELD_TYPES = {
    "n": pl.Int64,
    "operation": pl.String,
    "implementation": pl.String,
    "destination": pl.String,
    "cache_state": pl.String,
}


def main(
    run: Annotated[
        str, typer.Argument(help="Run directory under data/, e.g. cpu_sweep.")
    ],
) -> None:
    data_dir = DATA_ROOT / run
    archives = sorted(data_dir.glob("*.npz"))
    if len(archives) != 1:
        raise typer.BadParameter(
            f"Expected exactly one .npz in {data_dir}, found {len(archives)}."
        )

    data = np.load(archives[0])
    if "__meta__" not in data:
        raise ValueError(f"__meta__ not found in {archives[0]}")

    timings: dict[str, np.ndarray] = {
        key: data[key] for key in data.files if key != "__meta__"
    }
    if not timings:
        raise ValueError(f"No timing arrays in {archives[0]}")

    widths = {len(key.split("__")) for key in timings}
    if widths not in ({len(FIELDS)}, {len(FIELDS) - 1}):
        raise ValueError(f"Unexpected timing-key widths: {sorted(widths)}")

    # Runs at a single size drop the leading n<size> field.
    fields = FIELDS if widths == {len(FIELDS)} else FIELDS[1:]

    if fields[0] == "n" and not all(key.startswith("n") for key in timings):
        raise ValueError("Expected every timing key to start with n<size>.")
    if len({samples.shape for samples in timings.values()}) != 1:
        raise ValueError("Timing arrays do not all have the same shape.")
    if (first := next(iter(timings.values()))).ndim != 1:
        raise ValueError("Timing arrays are not one-dimensional.")

    repeats = len(first)

    (data_dir / f"{run}.json").write_text(
        json.dumps(json.loads(data["__meta__"].item()), indent=2, sort_keys=True) + "\n"
    )

    levels = [
        sorted({key.split("__")[i] for key in timings}) for i in range(len(fields))
    ]

    rows = []
    for combo in itertools.product(*levels):
        samples = timings.get("__".join(combo))
        values = (int(combo[0][1:]), *combo[1:]) if fields[0] == "n" else combo
        for sample_index in range(repeats):
            time_us = None if samples is None else float(samples[sample_index])
            rows.append((*values, sample_index, time_us))

    table = pl.DataFrame(
        rows,
        schema={
            **{field: FIELD_TYPES[field] for field in fields},
            "sample_index": pl.Int16,
            "time_us": pl.Float64,
        },
        orient="row",
    ).sort(*fields, "sample_index")

    parquet_path = data_dir / f"{run}.parquet"
    table.write_parquet(parquet_path, compression="zstd")

    missing = len(table) // repeats - len(timings)
    print(f"Wrote {table.height:,} rows to {parquet_path}")
    print(
        f"Missing combinations: {missing:,} ({table['time_us'].null_count():,} nulls)"
    )


if __name__ == "__main__":
    typer.run(main)
