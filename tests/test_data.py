"""Loading and derived-metric checks for ``analysis.data``."""

import polars as pl

from analysis import data


def test_every_primary_run_is_loaded(d):
    assert set(data.RUNS) <= set(d.runs)


def test_pure_python_run_is_aggregated_per_cell(d):
    assert set(d.pure["implementation"].unique()) == {
        "py-list-float",
        "py-list-f32",
        "py-obj-array",
    }
    assert d.pure["samples"].min() == 30
    largest = d.pure.filter(pl.col("n") == 1_000_000)
    assert largest["n_elem"].unique().to_list() == [4_000_000]


def test_pure_python_memory_is_bytes_per_element(d):
    assert set(d.pure_memory) == {"py-list-float", "py-list-f32", "py-obj-array"}
    assert 47 < d.pure_memory["py-list-float"] < 49
