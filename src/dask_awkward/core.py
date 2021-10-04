from __future__ import annotations

import functools
import operator
from numbers import Number
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import awkward as ak
import numpy as np
from dask.base import DaskMethodsMixin, replace_name_in_key, tokenize
from dask.blockwise import blockwise as core_blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.threaded import get as threaded_get
from dask.utils import key_split, IndexCallable


def _finalize_daskawkwardarray(results: Any) -> Any:
    if all(isinstance(r, ak.Array) for r in results):
        return ak.concatenate(results)
    if all(isinstance(r, ak.Record) for r in results):
        raise NotImplementedError("Records not supported yet.")
    else:
        return results


def _finalize_scalar(results: Any) -> Any:
    return results[0]


class Scalar(DaskMethodsMixin):
    def __init__(self, dsk: HighLevelGraph, key: str) -> None:
        self._dask: HighLevelGraph = dsk
        self._key: str = key

    def __dask_graph__(self):
        return self._dask

    def __dask_keys__(self):
        return [self._key]

    def __dask_layers__(self):
        if isinstance(self._dask, HighLevelGraph) and len(self._dask.layers) == 1:
            return tuple(self._dask.layers)
        return (self.key,)

    def __dask_tokenize__(self):
        return self.key

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    __dask_scheduler__ = staticmethod(threaded_get)

    def __dask_postcompute__(self):
        return _finalize_scalar, ()

    def __dask_postpersist__(self):
        return self._rebuild, ()

    def _rebuild(self, dsk, *, rename=None):
        key = replace_name_in_key(self.key, rename) if rename else self.key
        return Scalar(dsk, key)

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self.key

    def __add__(self, other: Scalar) -> Scalar:
        name = "add-{}".format(tokenize(self, other))
        deps = [self, other]
        llg = {name: (operator.add, self.key, other.key)}
        g = HighLevelGraph.from_collections(name, llg, dependencies=deps)
        return new_scalar_object(g, name, None)


def new_scalar_object(dsk: HighLevelGraph, name: str, meta: Any):
    return Scalar(dsk, name)


class DaskAwkwardArray(DaskMethodsMixin):
    """Partitioned, lazy, and parallel Awkward Array Dask collection.

    The class constructor is not intended for users. Instead use
    factory functions like :py:func:`dask_awkward.from_parquet,
    :py:func:`dask_awkward.from_json`, etc.

    Within dask-awkward the ``new_array_object`` factory function is
    used for creating new instances.

    """

    def __init__(
        self,
        dsk: HighLevelGraph,
        key: str,
        divisions: Tuple[Any, ...] = None,
        npartitions: int = None,
    ) -> None:
        self._dask: HighLevelGraph = dsk
        self._key: str = key
        if divisions is None and npartitions is not None:
            self._npartitions: int = npartitions
            self._divisions: Tuple[Any, ...] = (None,) * (npartitions + 1)
        elif divisions is not None and npartitions is None:
            self._divisions = divisions
            self._npartitions = len(divisions) - 1
        self._fields: List[str] = None

    def __dask_graph__(self) -> HighLevelGraph:
        return self.dask

    def __dask_keys__(self) -> List[Tuple[str, int]]:
        return [(self.name, i) for i in range(self.npartitions)]

    def __dask_layers__(self) -> Tuple[str]:
        return (self.name,)

    def __dask_tokenize__(self) -> str:
        return self.name

    def __dask_postcompute__(self) -> Any:
        return _finalize_daskawkwardarray, ()

    @staticmethod
    def __dask_optimize__(dsk, keys, **kwargs):
        return dsk

    def _rebuild(self, dsk: Any, *, rename: Any = None) -> Any:
        name = self.name
        if rename:
            name = rename.get(name, name)
        return type(self)(dsk, name, self.npartitions)

    def __str__(self) -> str:
        return (
            f"DaskAwkwardArray<{key_split(self.name)}, npartitions={self.npartitions}>"
        )

    __repr__ = __str__
    __dask_scheduler__ = staticmethod(threaded_get)

    @property
    def dask(self) -> HighLevelGraph:
        return self._dask

    @property
    def fields(self) -> Iterable[str]:
        return self._fields

    @property
    def key(self) -> str:
        return self._key

    @property
    def name(self) -> str:
        return self.key

    @property
    def divisions(self) -> Tuple[Any, ...]:
        return self._divisions

    @property
    def known_divisions(self) -> bool:
        return len(self.divisions) > 0 and self.divisions[0] is not None

    @property
    def npartitions(self) -> int:
        return self._npartitions

    def _partitions(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        token = tokenize(self, index)
        from dask.array.slicing import normalize_index

        index = normalize_index(index, (self.npartitions,))
        index = tuple(slice(k, k + 1) if isinstance(k, Number) else k for k in index)
        name = f"partitions-{token}"
        new_keys = np.array(self.__dask_keys__(), dtype=object)[index].tolist()
        print(f"{new_keys=}")
        divisions = [self.divisions[i] for _, i in new_keys] + [
            self.divisions[new_keys[-1][1] + 1]
        ]
        print(f"{self.divisions=}")
        print(f"{divisions=}")
        dsk = {(name, i): tuple(key) for i, key in enumerate(new_keys)}
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
        return new_array_object(graph, name, None, divisions=tuple(divisions))

    @property
    def partitions(self) -> IndexCallable:
        return IndexCallable(self._partitions)

    def __getitem__(self, key) -> Any:
        if not isinstance(key, (int, str)):
            raise NotImplementedError(
                "getitem supports only string and integer for now."
            )
        token = tokenize(self, key)
        name = f"getitem-{token}"
        graphlayer = pw_layer(
            lambda x, gikey: operator.getitem(x, gikey), name, self, gikey=key
        )
        hlg = HighLevelGraph.from_collections(name, graphlayer, dependencies=[self])
        return new_array_object(hlg, name, None, self.npartitions)

    def __getattr__(self, attr) -> Any:
        return self.__getitem__(attr)


def new_array_object(
    dsk: HighLevelGraph,
    name: str,
    meta: Any,
    npartitions: int = None,
    divisions: Tuple[Any, ...] = None,
):
    return DaskAwkwardArray(dsk, name, npartitions=npartitions, divisions=divisions)


def pw_layer(func, name, *args, **kwargs):
    pairs: List[Any] = []
    numblocks: Dict[Any, int] = {}
    for arg in args:
        if isinstance(arg, DaskAwkwardArray):
            pairs.extend([arg.name, "i"])
            numblocks[arg.name] = (arg.npartitions,)
    return core_blockwise(
        func,
        name,
        "i",
        *pairs,
        numblocks=numblocks,
        concatenate=True,
        **kwargs,
    )


def pw_reduction_with_agg(
    a: DaskAwkwardArray,
    func: Callable,
    agg: Callable,
    *,
    name: str = None,
    **kwargs,
):
    token = tokenize(a)
    name = func.__name__ if name is None else name
    name = f"{name}-{token}"
    func = functools.partial(func, **kwargs)
    dsk = {(name, i): (func, k) for i, k in enumerate(a.__dask_keys__())}
    dsk[name] = (agg, list(dsk.keys()))
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[a])
    return new_scalar_object(hlg, name, None)


class TrivialPartitionwiseOp:
    def __init__(self, func: Callable, name: str = None) -> None:
        self._func = func
        self.__name__ = func.__name__ if name is None else name

    def __call__(self, collection, **kwargs):
        token = tokenize(collection)
        name = f"{self.__name__}-{token}"
        layer = pw_layer(self._func, name, collection, **kwargs)
        hlg = HighLevelGraph.from_collections(name, layer, dependencies=[collection])
        return new_array_object(hlg, name, None, collection.npartitions)


_count_trivial = TrivialPartitionwiseOp(ak.count)
_flatten_trivial = TrivialPartitionwiseOp(ak.flatten)
_max_trivial = TrivialPartitionwiseOp(ak.max)
_min_trivial = TrivialPartitionwiseOp(ak.min)
_num_trivial = TrivialPartitionwiseOp(ak.num)
_sum_trivial = TrivialPartitionwiseOp(ak.sum)


def count(a, axis: Optional[int] = None, **kwargs):
    if axis is not None and axis > 0:
        return _count_trivial(a, axis=axis, **kwargs)
    elif axis is None:
        trivial_result = _count_trivial(a, axis=1, **kwargs)
        return pw_reduction_with_agg(trivial_result, ak.sum, ak.sum)
    elif axis == 0 or axis == -1 * a.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


def flatten(a: DaskAwkwardArray, axis: int = 1, **kwargs):
    if axis > 0:
        return _flatten_trivial(a, axis=axis, **kwargs)
    raise NotImplementedError(f"axis={axis} is not supported for this array yet.")


def max(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    if axis == 1:
        return _max_trivial(a, axis=axis, **kwargs)
    elif axis is None:
        trivial_result = _max_trivial(a, axis=1, **kwargs)
        return pw_reduction_with_agg(trivial_result, ak.max, ak.max, **kwargs)
    elif axis == 0 or axis == -1 * a.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


def min(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    if axis == 1:
        return _min_trivial(a, axis=axis, **kwargs)
    elif axis is None:
        trivial_result = _min_trivial(a, axis=1, **kwargs)
        return pw_reduction_with_agg(trivial_result, ak.min, ak.min, **kwargs)
    elif axis == 0 or axis == -1 * a.ndim:
        raise NotImplementedError(f"axis={axis} is not supported for this array yet.")
    else:
        raise ValueError("axis must be None or an integer.")


def num(a: DaskAwkwardArray, axis: int = 1, **kwargs):
    pass


def sum(a: DaskAwkwardArray, axis: Optional[int] = None, **kwargs):
    pass
