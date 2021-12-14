from __future__ import annotations

from typing import Any

import numpy as np
from dask.base import is_dask_collection

from .core import Scalar


def assert_eq(a: Any, b: Any) -> None:
    if is_dask_collection(a) and not is_dask_collection(b):
        if isinstance(a, Scalar):
            assert a.compute() == b
        else:
            assert a.compute().to_list() == b.to_list()
    elif is_dask_collection(b) and not is_dask_collection(a):
        if isinstance(b, Scalar):
            assert a == b.compute()
        else:
            assert a.to_list() == b.compute().to_list()
    else:
        if isinstance(a, Scalar) and isinstance(b, Scalar):
            assert a.compute() == b.compute()
        else:
            assert a.compute().to_list() == b.compute().to_list()


def normalize_single_outer_inner_index(
    divisions: tuple[int, ...], index: int
) -> tuple[int, int]:
    """Determine partition index and inner index for some divisions.

    Parameters
    ----------
    divisions : tuple[int, ...]
        The divisions of a Dask awkward collection.
    index : int
        The overall index (for the complete collection).

    Returns
    -------
    int
        Which partition in the collection.
    int
        Which inner index in the determined partition.

    Examples
    --------
    >>> from dask_awkward.utils import normalize_single_outer_inner_index
    >>> divisions = (0, 3, 6, 9)
    >>> normalize_single_outer_inner_index(divisions, 0)
    (0, 0)
    >>> normalize_single_outer_inner_index(divisions, 5)
    (1, 2)
    >>> normalize_single_outer_inner_index(divisions, 8)
    (2, 2)

    """
    if len(divisions) == 2:
        return (0, index)
    partition_index = int(np.digitize(index, divisions)) - 1
    new_index = index - divisions[partition_index]
    return (partition_index, new_index)
