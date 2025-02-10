from __future__ import annotations

import builtins
import warnings
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from numbers import Number
from typing import TYPE_CHECKING, Any

import awkward as ak
import numpy as np
from awkward.types.type import Type
from awkward.typetracer import create_unknown_scalar, is_unknown_scalar
from dask.base import is_dask_collection, tokenize, unpack_collections
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers import AwkwardMaterializedLayer
from dask_awkward.lib.core import (
    Array,
    PartitionCompatibility,
    _map_partitions,
    map_partitions,
    new_known_scalar,
    new_scalar_object,
    partition_compatibility,
)
from dask_awkward.utils import (
    DaskAwkwardNotImplemented,
    IncompatiblePartitions,
    borrow_docstring,
    first,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


__all__ = (
    "argcartesian",
    "argcombinations",
    "argsort",
    "broadcast_arrays",
    "cartesian",
    "combinations",
    "copy",
    "drop_none",
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
    "pad_none",
    "ravel",
    "run_lengths",
    "singletons",
    "sort",
    "strings_astype",
    "to_list",
    "to_packed",
    "to_regular",
    "unflatten",
    "unzip",
    "values_astype",
    "where",
    "with_field",
    "without_field",
    "with_name",
    "with_parameter",
    "without_parameters",
    "zeros_like",
    "zip",
    "zip_no_broadcast",
)


class _ArgCartesianFn:
    def __init__(self, repacker, **kwargs):
        self.repacker = repacker
        self.kwargs = kwargs

    def __call__(self, *arrays):
        return ak.argcartesian(self.repacker(arrays)[0], **self.kwargs)


@borrow_docstring(ak.argcartesian)
def argcartesian(
    arrays: Sequence[Array] | Mapping[str, Array],
    axis: int = 1,
    nested: bool | Iterable[str | int] | None = None,
    parameters: dict[str, Any] | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    # FIXME: resolve negative axis
    if axis >= 1:
        arrays_unpacked, repacker = unpack_collections(arrays, traverse=True)
        fn = _ArgCartesianFn(
            repacker,
            axis=axis,
            nested=nested,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )
        return map_partitions(
            fn, *arrays_unpacked, label="argcartesian", output_divisions=1
        )
    raise DaskAwkwardNotImplemented("TODO")


class _ArgCombinationsFn:
    def __init__(self, n: int, axis: int, **kwargs: Any):
        self.n = n
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.argcombinations(array, self.n, axis=self.axis, **self.kwargs)


@borrow_docstring(ak.argcombinations)
def argcombinations(
    array: Array,
    n: int,
    replacement: bool = False,
    axis: int = 1,
    fields: Sequence[str] | None = None,
    parameters: dict[str, Any] | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if fields is not None and len(fields) != n:
        raise ValueError("if provided, the length of 'fields' must be 'n'")

    # FIXME: resolve negative axis
    if axis < 0:
        raise ValueError("the 'axis' for argcombinations must be non-negative")

    if axis >= 0:
        fn = _ArgCombinationsFn(
            n=n,
            replacement=replacement,
            axis=axis,
            fields=fields,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )
        return map_partitions(
            fn,
            array,
            label="argcombinations",
            output_divisions=1,
        )
    raise DaskAwkwardNotImplemented("TODO")


class _ArgsortFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.argsort(array, **self.kwargs)


@borrow_docstring(ak.argsort)
def argsort(
    array: Array,
    axis: int = -1,
    ascending: bool = True,
    stable: bool = True,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if axis == 0:
        raise DaskAwkwardNotImplemented("TODO")
    fn = _ArgsortFn(
        axis=axis, ascending=ascending, stable=stable, behavior=behavior, attrs=attrs
    )
    return map_partitions(fn, array, label="argsort", output_divisions=1)


class _BroadcastArraysFn:
    def __init__(self, index, **kwargs):
        self.index = index
        self.kwargs = kwargs

    def __call__(self, *arrays):
        return ak.broadcast_arrays(*arrays, **self.kwargs)[self.index]


@borrow_docstring(ak.broadcast_arrays)
def broadcast_arrays(
    *arrays: Array, highlevel: bool = True, **kwargs: Any
) -> list[Array]:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if partition_compatibility(*arrays) == PartitionCompatibility.NO:
        raise IncompatiblePartitions("broadcast_arrays", *arrays)

    array_metas = (array._meta for array in arrays)

    metas = ak.broadcast_arrays(*array_metas, highlevel=highlevel, **kwargs)

    # here we return the list of broadcasted arrays
    # it's OK to repeat the work this way since usually
    # only one of the outputs will be computed, and
    # broadcast_arrays is fast anyway
    return [
        map_partitions(
            _BroadcastArraysFn(i, highlevel=highlevel, **kwargs),
            *arrays,
            label="broadcast-arrays",
            meta=meta,
        )
        for i, meta in enumerate(metas)
    ]


class _CartesianFn:
    def __init__(self, repacker, **kwargs):
        self.repacker = repacker
        self.kwargs = kwargs

    def __call__(self, *arrays):
        return ak.cartesian(self.repacker(arrays)[0], **self.kwargs)


@borrow_docstring(ak.cartesian)
def cartesian(
    arrays: Sequence[Array] | Mapping[str, Array],
    axis: int = 1,
    nested: bool | Iterable[str | int] | None = None,
    parameters: dict[str, Any] | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if axis == 1:
        arrays_unpacked, repacker = unpack_collections(arrays, traverse=True)
        fn = _CartesianFn(
            repacker,
            axis=axis,
            nested=nested,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )
        return map_partitions(
            fn, *arrays_unpacked, label="cartesian", output_divisions=1
        )
    raise DaskAwkwardNotImplemented("TODO")


class _CombinationsFn:
    def __init__(self, n: int, axis: int, **kwargs: Any):
        self.n = n
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.combinations(array, self.n, axis=self.axis, **self.kwargs)


@borrow_docstring(ak.combinations)
def combinations(
    array: Array,
    n: int,
    replacement: bool = False,
    axis: int = 1,
    fields: list[str] | None = None,
    parameters: Mapping[str, Any] | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if fields is not None and len(fields) != n:
        raise ValueError("if provided, the length of 'fields' must be 'n'")

    if axis != 0:
        fn = _CombinationsFn(
            n=n,
            replacement=replacement,
            axis=axis,
            fields=fields,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )
        return map_partitions(
            fn,
            array,
            label="combinations",
            output_divisions=1,
        )
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.copy)
def copy(array: Array) -> Array:
    # Make a copy of meta, but don't try and copy the layout;
    # dask-awkward's copy is metadata-only
    old_meta = array._meta
    new_meta = ak.Array(old_meta.layout, behavior=deepcopy(old_meta._behavior))

    return Array(
        array._dask,
        array._name,
        new_meta,
        array._divisions,
    )


class _FillNoneFn:
    def __init__(self, value, **kwargs):
        self.value = value
        self.kwargs = kwargs

    def __call__(self, arr):
        return ak.fill_none(arr, self.value, **self.kwargs)


@borrow_docstring(ak.fill_none)
def fill_none(
    array: Array,
    value: Any,
    axis: int | None = -1,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    fn = _FillNoneFn(
        value, axis=axis, highlevel=highlevel, behavior=behavior, attrs=attrs
    )
    return map_partitions(fn, array, label="fill-none", output_divisions=1)


class _DropNoneFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, arr):
        return ak.drop_none(arr, **self.kwargs)


@borrow_docstring(ak.drop_none)
def drop_none(
    array: Array,
    axis: int | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    fn = _DropNoneFn(axis=axis, highlevel=highlevel, behavior=behavior, attrs=attrs)
    return map_partitions(fn, array, label="drop-none", output_divisions=1)


class _FirstsFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.firsts(array, **self.kwargs)


@borrow_docstring(ak.firsts)
def firsts(
    array: Array,
    axis: int = 1,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if axis >= 1:
        return map_partitions(
            _FirstsFn(axis=axis, highlevel=highlevel, behavior=behavior, attrs=attrs),
            array,
            label="firsts",
            output_divisions=1,
        )
    elif axis == 0:
        return array[0]
    raise DaskAwkwardNotImplemented("TODO")


class _FlattenFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array: ak.Array) -> ak.Array:
        return ak.flatten(array, **self.kwargs)


@borrow_docstring(ak.flatten)
def flatten(
    array: Array,
    axis: int | None = 1,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        _FlattenFn(axis=axis, highlevel=highlevel, behavior=behavior, attrs=attrs),
        array,
        label="flatten",
        output_divisions=None,
    )


@borrow_docstring(ak.from_regular)
def from_regular(
    array: Array,
    axis: int = 1,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if axis == 0:
        raise ValueError("axis must be > 0 for from_regular")

    return map_partitions(
        ak.from_regular,
        array,
        axis=axis,
        highlevel=highlevel,
        behavior=behavior,
        label="from-regular",
        attrs=attrs,
    )


@borrow_docstring(ak.full_like)
def full_like(
    array: Array,
    fill_value: Any,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    dtype: DTypeLike | str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if dtype is str:
        raise ValueError(
            """dtype cannot be 'str' for dak.full_like,
            you can accomplish this with dask-array and
            dak.flatten/dak.unflatten"""
        )

    return map_partitions(
        ak.full_like,
        array,
        fill_value,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
        dtype=dtype,
        output_divisions=1,
    )


@borrow_docstring(ak.isclose)
def isclose(
    a: Array,
    b: Array,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if partition_compatibility(a, b) == PartitionCompatibility.NO:
        raise IncompatiblePartitions("isclose", a, b)

    return map_partitions(
        ak.isclose,
        a,
        b,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        highlevel=highlevel,
        behavior=behavior,
        label="is-close",
        output_divisions=1,
        attrs=attrs,
    )


class _IsNoneFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.is_none(array, **self.kwargs)


@borrow_docstring(ak.is_none)
def is_none(
    array: Array,
    axis: int = 0,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    fn = _IsNoneFn(axis=axis, highlevel=highlevel, behavior=behavior, attrs=attrs)
    return map_partitions(fn, array, label="is-none", output_divisions=1)


@borrow_docstring(ak.local_index)
def local_index(
    array: Array,
    axis: int = -1,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if axis <= 0:
        DaskAwkwardNotImplemented("axis<=0 for local_index is not supported")
    return map_partitions(
        ak.local_index,
        array,
        axis=axis,
        highlevel=highlevel,
        behavior=behavior,
        attrs=attrs,
    )


@borrow_docstring(ak.mask)
def mask(
    array: Array,
    mask: Array,
    valid_when: bool = True,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if partition_compatibility(array, mask) == PartitionCompatibility.NO:
        raise IncompatiblePartitions("mask", array, mask)
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        ak.mask, array, mask, valid_when=valid_when, behavior=behavior, attrs=attrs
    )


@borrow_docstring(ak.nan_to_num)
def nan_to_num(
    array: Array,
    nan: float = 0.0,
    posinf: Any | None = None,
    neginf: Any | None = None,
    behavior: Any | None = None,
) -> Array:
    return map_partitions(
        ak.nan_to_num,
        array,
        nan=nan,
        posinf=posinf,
        neginf=neginf,
        highlevel=True,
        behavior=behavior,
    )


def _numaxis0(*integers):
    f = first(integers)
    if is_unknown_scalar(f):
        return f
    return np.sum(np.array(integers))


@borrow_docstring(ak.num)
def num(
    array: Any,
    axis: int = 1,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Any:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if axis == 0 or axis == -1 * array.ndim:
        if array.known_divisions:
            return new_known_scalar(array.defined_divisions[-1], label="num")

        per_axis = map_partitions(
            ak.num,
            array,
            axis=0,
            meta=ak.Array(ak.Array([1, 1]).layout.to_typetracer(forget_length=True)),
        )
        name = f"numaxis0-{tokenize(array, axis)}"
        keys = per_axis.__dask_keys__()
        matlayer = AwkwardMaterializedLayer(
            {(name, 0): (_numaxis0, *keys)}, previous_layer_names=[per_axis.name]
        )
        hlg = HighLevelGraph.from_collections(name, matlayer, dependencies=(per_axis,))
        return new_scalar_object(
            hlg,
            name,
            meta=ak.Array(create_unknown_scalar(np.dtype("int64"))),
        )
    else:
        return map_partitions(
            ak.num,
            array,
            axis=axis,
            behavior=behavior,
            output_divisions=1,
            attrs=attrs,
        )


@borrow_docstring(ak.ones_like)
def ones_like(
    array: Array,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
    dtype: DTypeLike | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        ak.ones_like,
        array,
        label="ones-like",
        behavior=behavior,
        dtype=dtype,
        output_divisions=1,
        attrs=attrs,
    )


@borrow_docstring(ak.to_list)
def to_list(array: Array) -> list:
    """Return list/dict version of the data

    Unlike most functions, this one requires a compute() of the data.
    """
    return array.compute().to_list()


@borrow_docstring(ak.to_packed)
def to_packed(
    array: Array,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    return map_partitions(ak.to_packed, array, behavior=behavior, attrs=attrs)


class _PadNoneFn:
    def __init__(self, target, axis, **kwargs):
        self.target = target
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.pad_none(
            array,
            target=self.target,
            axis=self.axis,
            **self.kwargs,
        )


@borrow_docstring(ak.pad_none)
def pad_none(
    array: Array,
    target: bool,
    axis: int = 1,
    clip: bool = False,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if axis == 0:
        DaskAwkwardNotImplemented("axis=0 for pad_none is not supported")
    return map_partitions(
        _PadNoneFn(target=target, axis=axis, clip=clip, behavior=behavior, attrs=attrs),
        array,
        label="pad-none",
        output_divisions=1,
    )


@borrow_docstring(ak.ravel)
def ravel(
    array: Array,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if isinstance(array._meta.layout, ak.contents.recordarray.RecordArray):
        warnings.warn("ravel may produce inconsistent results for record arrays!")

    return map_partitions(
        ak.ravel,
        array,
        behavior=behavior,
        attrs=attrs,
        label="ravel",
    )


@borrow_docstring(ak.run_lengths)
def run_lengths(
    array: Array,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    # TODO: fix incorrect results for run_lengths when one dimensional
    minmax_depth = array._meta.layout.minmax_depth
    if minmax_depth[0] == 1 or minmax_depth[1] == 1:
        warnings.warn(
            "run_lengths can produce incorrect results for one dimensional arrays!"
        )

    return map_partitions(
        ak.run_lengths,
        array,
        behavior=behavior,
        attrs=attrs,
        label="run-lengths",
    )


class _SingletonsFn:
    def __init__(self, axis, **kwargs):
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.singletons(array, axis=self.axis, **self.kwargs)


@borrow_docstring(ak.singletons)
def singletons(
    array: Array,
    axis: int = 0,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    return map_partitions(
        _SingletonsFn(axis, behavior=behavior, attrs=attrs),
        array,
        label="singletons",
    )


class _SortFn:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, array):
        return ak.sort(array, **self.kwargs)


@borrow_docstring(ak.sort)
def sort(
    array: Array,
    axis: int = -1,
    ascending: bool = True,
    stable: bool = True,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    if axis == 0:
        raise DaskAwkwardNotImplemented("TODO")
    fn = _SortFn(
        axis=axis,
        ascending=ascending,
        stable=stable,
        behavior=behavior,
        attrs=attrs,
    )
    return map_partitions(fn, array, label="sort", output_divisions=1)


@borrow_docstring(ak.strings_astype)
def strings_astype(
    array: Array,
    to: np.dtype | str,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    raise DaskAwkwardNotImplemented("TODO")


@borrow_docstring(ak.to_regular)
def to_regular(
    array: Array,
    axis: int = 1,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if axis == 0:
        raise ValueError("axis must be > 0 for from_regular")

    #  NB: It is impossible to compute the typetracer for this.
    #      We don't know the output array size in general,
    #      since it is var.
    return map_partitions(
        ak.to_regular,
        array,
        axis=axis,
        behavior=behavior,
        label="to-regular",
        attrs=attrs,
    )


@borrow_docstring(ak.unflatten)
def unflatten(
    array: Array,
    counts: int | Array,
    axis: int = 0,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    warnings.warn(
        f"""Please ensure that {counts}
        is partitionwise-compatible with {array}
        (e.g. counts comes from a dak.num(array, axis=1)),
        otherwise this unflatten operation will fail when computed!"""
    )

    return map_partitions(
        ak.unflatten,
        array,
        counts,
        axis=axis,
        behavior=behavior,
        label="unflatten",
    )


def _array_with_rebuilt_meta(
    array: Array, behavior: Mapping | None, attrs: Mapping[str, Any] | None
) -> Array:
    if attrs is None:
        attrs = array._meta.attrs

    if behavior is None:
        behavior = array._meta.behavior

    new_meta = ak.Array(array._meta, behavior=behavior, attrs=attrs)

    return Array(array.dask, array.name, new_meta, array.divisions)


@borrow_docstring(ak.unzip)
def unzip(
    array: Array,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> tuple[Array, ...]:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    fields = ak.fields(array._meta)
    if len(fields) == 0:
        return (_array_with_rebuilt_meta(array, behavior, attrs),)
    else:
        return tuple(
            _array_with_rebuilt_meta(array[field], behavior, attrs) for field in fields
        )


@borrow_docstring(ak.values_astype)
def values_astype(
    array: Array,
    to: np.dtype | str,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        ak.values_astype,
        array,
        to=to,
        behavior=behavior,
        label="values-astype",
        attrs=attrs,
    )


class _WhereFn:
    def __init__(
        self,
        mergebool: bool = True,
        highlevel: bool = True,
        behavior: Mapping | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.mergebool = mergebool
        self.highlevel = highlevel
        self.behavior = behavior
        self.attrs = attrs

    def __call__(self, condition: ak.Array, x: ak.Array, y: ak.Array) -> ak.Array:
        return ak.where(
            condition,
            x,
            y,
            mergebool=self.mergebool,
            highlevel=self.highlevel,
            behavior=self.behavior,
            attrs=self.attrs,
        )


@borrow_docstring(ak.where)
def where(
    condition: Array,
    x: Array,
    y: Array,
    mergebool: bool = True,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    maybe_dask_args = [condition, x, y]
    dask_args = tuple(arg for arg in maybe_dask_args if is_dask_collection(arg))

    if not isinstance(condition, Array):
        raise ValueError(
            "The condition argugment to where must be a dask_awkward.Array"
        )

    if partition_compatibility(*dask_args) == PartitionCompatibility.NO:
        raise IncompatiblePartitions("where", *dask_args)

    return map_partitions(
        _WhereFn(mergebool=mergebool, behavior=behavior, attrs=attrs),
        condition,
        x,
        y,
        label="where",
    )


class _WithFieldFn:
    def __init__(
        self,
        where: str | Sequence[str] | None,
        highlevel: bool,
        behavior: Mapping | None,
        attrs: Mapping[str, Any] | None,
    ) -> None:
        self.where = where
        self.highlevel = highlevel
        self.behavior = behavior
        self.attrs = attrs

    def __call__(self, base: ak.Array, what: ak.Array) -> ak.Array:
        return ak.with_field(
            base, what, where=self.where, behavior=self.behavior, attrs=self.attrs
        )


@borrow_docstring(ak.with_field)
def with_field(
    base: Array,
    what: Array | int | float | complex | bool,
    where: str | Sequence[str] | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if not isinstance(base, Array):
        raise ValueError("Base argument in with_field must be a dask_awkward.Array")

    if not isinstance(what, (Array, Number)):
        raise ValueError(
            "with_field cannot accept string, bytes, list, or dict values yet"
        )

    maybe_dask_args = [base, what]
    dask_args = tuple(arg for arg in maybe_dask_args if is_dask_collection(arg))

    if partition_compatibility(*dask_args) == PartitionCompatibility.NO:
        raise IncompatiblePartitions("with_field", *dask_args)
    return _map_partitions(
        _WithFieldFn(where=where, highlevel=highlevel, behavior=behavior, attrs=attrs),
        base,
        what,
        label="with-field",
        output_divisions=1,
    )


class _WithoutFieldFn:
    def __init__(
        self,
        highlevel: bool,
        behavior: Mapping | None = None,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self.highlevel = highlevel
        self.behavior = behavior
        self.attrs = attrs

    def __call__(self, array: ak.Array, where: str) -> ak.Array:
        return ak.without_field(
            array, where=where, behavior=self.behavior, attrs=self.attrs
        )


@borrow_docstring(ak.without_field)
def without_field(
    base: Array,
    where: str,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if not isinstance(base, Array):
        raise ValueError("Base argument in without_field must be a dask_awkward.Array")

    return map_partitions(
        _WithoutFieldFn(highlevel=highlevel, behavior=behavior, attrs=attrs),
        base,
        where,
        label="without-field",
        output_divisions=1,
    )


class _WithNameFn:
    def __init__(
        self,
        name: str | None,
        behavior: Mapping | None,
        attrs: Mapping[str, Any] | None,
    ) -> None:
        self.name = name
        self.behavior = behavior
        self.attrs = attrs

    def __call__(self, array: ak.Array) -> ak.Array:
        return ak.with_name(array, self.name, behavior=self.behavior, attrs=self.attrs)


@borrow_docstring(ak.with_name)
def with_name(
    array: Array,
    name: str | None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    return map_partitions(
        _WithNameFn(name=name, behavior=behavior, attrs=attrs),
        array,
        label="with-name",
        output_divisions=1,
    )


class _WithParameterFn:
    def __init__(
        self,
        parameter: str,
        value: Any,
        behavior: Mapping | None,
        attrs: Mapping[str, Any] | None,
    ):
        self.parameter = parameter
        self.value = value
        self.behavior = behavior
        self.attrs = attrs

    def __call__(self, array):
        return ak.with_parameter(
            array,
            parameter=self.parameter,
            value=self.value,
            behavior=self.behavior,
            attrs=self.attrs,
        )


@borrow_docstring(ak.with_parameter)
def with_parameter(
    array: Array,
    parameter: str,
    value: Any,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        _WithParameterFn(
            parameter=parameter, value=value, behavior=behavior, attrs=attrs
        ),
        array,
        label="with-parameter",
        output_divisions=1,
    )


class _WithoutParameterFn:
    def __init__(self, behavior: Mapping | None, attrs: Mapping[str, Any] | None):
        self.behavior = behavior
        self.attrs = attrs

    def __call__(self, array):
        return ak.without_parameters(array, behavior=self.behavior, attrs=self.attrs)


@borrow_docstring(ak.without_parameters)
def without_parameters(
    array: Array,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        _WithoutParameterFn(behavior=behavior, attrs=attrs),
        array,
        label="without-parameters",
        output_divisions=1,
    )


@borrow_docstring(ak.zeros_like)
def zeros_like(
    array: Array,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
    dtype: DTypeLike | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")
    return map_partitions(
        ak.zeros_like,
        array,
        label="zeros-like",
        behavior=behavior,
        dtype=dtype,
        output_divisions=1,
        attrs=attrs,
    )


class _ZipDictInputFn:
    def __init__(self, keys: Sequence[str], **kwargs: Any) -> None:
        self.keys = keys
        self.kwargs = kwargs

    def __call__(self, *parts: ak.Array) -> ak.Array:
        return ak.zip(
            {k: p for k, p in builtins.zip(self.keys, list(parts))},
            **self.kwargs,
        )


class _ZipListInputFn:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, *parts: Any) -> ak.Array:
        return ak.zip(list(parts), **self.kwargs)


@borrow_docstring(ak.zip)
def zip(
    arrays: Sequence[Array] | Mapping[str, Array],
    depth_limit: int | None = None,
    parameters: Mapping[str, Any] | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    right_broadcast: bool = False,
    optiontype_outside_record: bool = False,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if isinstance(arrays, Mapping):
        keys, colls, metadict = [], [], {}
        for k, coll in arrays.items():
            keys.append(k)
            colls.append(coll)
            metadict[k] = coll._meta

        meta = ak.zip(
            metadict,
            depth_limit=depth_limit,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            right_broadcast=right_broadcast,
            optiontype_outside_record=optiontype_outside_record,
            attrs=attrs,
        )

        return map_partitions(
            _ZipDictInputFn(
                keys,
                depth_limit=depth_limit,
                parameters=parameters,
                with_name=with_name,
                highlevel=highlevel,
                behavior=behavior,
                right_broadcast=right_broadcast,
                optiontype_outside_record=optiontype_outside_record,
                attrs=attrs,
            ),
            *colls,
            label="zip",
            meta=meta,
        )

    elif isinstance(arrays, Sequence):
        fn = _ZipListInputFn(
            depth_limit=depth_limit,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            right_broadcast=right_broadcast,
            optiontype_outside_record=optiontype_outside_record,
            attrs=attrs,
        )
        return map_partitions(
            fn,
            *arrays,
            label="zip",
        )

    else:
        raise DaskAwkwardNotImplemented(
            "only mappings or sequences are supported by dak.zip (e.g. dict, list, or tuple)"
        )


class _ZipNoBroadcastDictInputFn:
    def __init__(self, keys: Sequence[str], **kwargs: Any) -> None:
        self.keys = keys
        self.kwargs = kwargs

    def __call__(self, *parts: ak.Array) -> ak.Array:
        return ak.zip_no_broadcast(
            {k: p for k, p in builtins.zip(self.keys, list(parts))},
            **self.kwargs,
        )


class _ZipNoBroadcastListInputFn:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, *parts: Any) -> ak.Array:
        return ak.zip_no_broadcast(list(parts), **self.kwargs)


@borrow_docstring(ak.zip_no_broadcast)
def zip_no_broadcast(
    arrays: Sequence[Array] | Mapping[str, Array],
    parameters: Mapping[str, Any] | None = None,
    with_name: str | None = None,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    if isinstance(arrays, Mapping):
        keys, colls, metadict = [], [], {}
        for k, coll in arrays.items():
            keys.append(k)
            colls.append(coll)
            metadict[k] = coll._meta

        meta = ak.zip_no_broadcast(
            metadict,
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )

        return map_partitions(
            _ZipNoBroadcastDictInputFn(
                keys,
                parameters=parameters,
                with_name=with_name,
                highlevel=highlevel,
                behavior=behavior,
                attrs=attrs,
            ),
            *colls,
            label="zip_no_broadcast",
            meta=meta,
        )

    elif isinstance(arrays, Sequence):
        fn = _ZipNoBroadcastListInputFn(
            parameters=parameters,
            with_name=with_name,
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )
        return map_partitions(
            fn,
            *arrays,
            label="zip_no_broadcast",
        )

    else:
        raise DaskAwkwardNotImplemented(
            "only mappings or sequences are supported by dak.zip_no_broadcast (e.g. dict, list, or tuple)"
        )


def _repartition_func(*stuff):
    import builtins

    import awkward as ak

    *data, slices = stuff
    data = [
        d[sl[0] : sl[1]] if sl is not None else d
        for d, sl in builtins.zip(data, slices)
    ]
    return ak.concatenate(data)


def repartition_layer(arr: Array, key: str, divisions: tuple[int, ...]) -> dict:
    layer = {}

    indivs = arr.defined_divisions
    i = 0
    for index, (start, end) in enumerate(builtins.zip(divisions[:-1], divisions[1:])):
        pp = []
        ss = []
        while indivs[i] <= start:
            i += 1
        j = i
        i -= 1
        while indivs[j] < end:
            j += 1
        for k in range(i, j):
            if start < indivs[k]:
                st = None
            elif start < indivs[k + 1]:
                st = start - indivs[k]
            else:
                continue
            if end < indivs[k]:
                continue
            elif end < indivs[k + 1]:
                en = end - indivs[k]
            else:
                en = None
            pp.append(k)
            ss.append((st, en))
        layer[(key, index)] = (
            (_repartition_func,) + tuple((arr.name, part) for part in pp) + (ss,)
        )
    return layer


def _subpart(data: ak.Array, parts: int, part: int) -> ak.Array:
    from dask_awkward.lib.core import is_typetracer

    if is_typetracer(data):
        return data
    rows_per = len(data) // parts
    return data[
        part * rows_per : None if part == (parts - 1) else (part + 1) * rows_per
    ]


def _subcat(*arrs: tuple[ak.Array, ...]) -> ak.Array:
    return ak.concatenate(arrs)


def simple_repartition_layer(
    arr: Array, n_to_one: int | None, one_to_n: int | None, key: str
) -> tuple[dict, tuple[Any, ...]]:
    layer: dict[tuple[str, int], tuple[Any, ...]] = {}
    new_divisions: tuple[Any, ...]
    if n_to_one:
        for i0, i in enumerate(range(0, arr.npartitions, n_to_one)):
            layer[(key, i0)] = (_subcat,) + tuple(
                (arr.name, part)
                for part in range(i, min(i + n_to_one, arr.npartitions))
            )
        new_divisions = arr.divisions[::n_to_one]
        if arr.npartitions % n_to_one:
            new_divisions = new_divisions + (arr.divisions[-1],)
            layer[(key, i0 + 1)] = (_subcat,) + tuple(
                (arr.name, part0) for part0 in range(len(layer), arr.npartitions)
            )
    elif one_to_n:
        for i in range(arr.npartitions):
            for part in range(one_to_n):
                layer[(key, (i * one_to_n + part))] = (
                    _subpart,
                    (arr.name, i),
                    one_to_n,
                    part,
                )
        # TODO: if arr.known_divisions:
        new_divisions = (None,) * (arr.npartitions * one_to_n + 1)
    else:
        raise ValueError
    return layer, new_divisions


@borrow_docstring(ak.enforce_type)
def enforce_type(
    array: Array,
    type: str | dict | Type,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    if not highlevel:
        raise ValueError("Only highlevel=True is supported")

    return map_partitions(
        ak.enforce_type,
        array,
        label="enforce-type",
        type=type,
        behavior=behavior,
        attrs=attrs,
        output_divisions=1,
    )
