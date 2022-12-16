from __future__ import annotations

import awkward as ak
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import assert_eq


@pytest.mark.parametrize("axis", [0])
def test_concatenate(daa, caa, axis):
    assert_eq(
        ak.concatenate([caa.points.x, caa.points.y], axis=axis),
        dak.concatenate([daa.points.x, daa.points.y], axis=axis),
    )
