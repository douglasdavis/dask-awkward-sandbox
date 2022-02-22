from __future__ import annotations

from typing import Any

import awkward._v2 as ak

from dask_awkward.core import DaskAwkwardNotImplemented, TrivialPartitionwiseOp

__all__ = (
    "argcartesian",
    "argcombinations",
    "argsort",
    "broadcast_arrays",
    "cartesian",
    "combinations",
    "concatenate",
    "copy",
    "fill_none",
    "firsts",
    "flatten",
    "from_regular",
    "full_like",
    "isclose",
    "is_none",
    "local_index",
    "mask",
    "nan_to_num",
    "num",
    "ones_like",
    "packed",
    "pad_none",
    "ravel",
    "run_lengths",
    "singletons",
    "sort",
    "strings_astype",
    "to_regular",
    "unflatten",
    "unzip",
    "values_astype",
    "where",
    "with_field",
    "with_name",
    "with_parameter",
    "without_parameters",
    "zeros_like",
    "zip",
)

_num_trivial = TrivialPartitionwiseOp(ak.num, axis=1)


def argcartesian(
    arrays,
    axis=1,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


def argcombinations(
    array,
    n,
    replacement=False,
    axis=1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


def argsort(array, axis=-1, ascending=True, stable=True, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def broadcast_arrays(*arrays, **kwargs):
    raise DaskAwkwardNotImplemented("TODO")


def cartesian(
    arrays,
    axis=1,
    nested=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


def combinations(
    array,
    n,
    replacement=False,
    axis=1,
    fields=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
):
    raise DaskAwkwardNotImplemented("TODO")


def concatenate(
    arrays, axis=0, merge=True, mergebool=True, highlevel=True, behavior=None
):
    raise DaskAwkwardNotImplemented("TODO")


def copy(array):
    raise DaskAwkwardNotImplemented("TODO")


def fill_none(array, value, axis=-1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def firsts(array, axis=1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def flatten(array, axis=1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def from_regular(array, axis=1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def full_like(array, fill_value, highlevel=True, behavior=None, dtype=None):
    raise DaskAwkwardNotImplemented("TODO")


def isclose(
    a, b, rtol=1e-05, atol=1e-08, equal_nan=False, highlevel=True, behavior=None
):
    raise DaskAwkwardNotImplemented("TODO")


def is_none(array, axis=0, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def local_index(array, axis=-1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def mask(array, mask, valid_when=True, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def nan_to_num(
    array, copy=True, nan=0.0, posinf=None, neginf=None, highlevel=True, behavior=None
):
    raise DaskAwkwardNotImplemented("TODO")


def num(array: Any, axis: int | None = 1, highlevel: bool = True, behavior=None) -> Any:
    if axis == 1:
        return _num_trivial(
            array,
            axis=axis,
            highlevel=True,
            behavior=behavior,
        )

    raise DaskAwkwardNotImplemented("TODO")


def ones_like(array, highlevel=True, behavior=None, dtype=None):
    raise DaskAwkwardNotImplemented("TODO")


def packed(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def pad_none(array, target, axis=1, clip=False, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def ravel(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def run_lengths(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def singletons(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def sort(array, axis=-1, ascending=True, stable=True, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def strings_astype(array, to, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def to_regular(array, axis=1, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def unflatten(array, counts, axis=0, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def unzip(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def values_astype(array, to, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def where(condition, *args, **kwargs):
    raise DaskAwkwardNotImplemented("TODO")


def with_field(base, what, where=None, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def with_name(array, name, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def with_parameter(array, parameter, value, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def without_parameters(array, highlevel=True, behavior=None):
    raise DaskAwkwardNotImplemented("TODO")


def zeros_like(array, highlevel=True, behavior=None, dtype=None):
    raise DaskAwkwardNotImplemented("TODO")


def zip(
    arrays,
    depth_limit=None,
    parameters=None,
    with_name=None,
    highlevel=True,
    behavior=None,
    right_broadcast=False,
):
    raise DaskAwkwardNotImplemented("TODO")
