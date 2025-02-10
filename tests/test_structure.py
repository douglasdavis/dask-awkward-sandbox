from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

import dask_awkward as dak
from dask_awkward.lib.testutils import AK_LTE_2_5_0, assert_eq
from dask_awkward.utils import DaskAwkwardNotImplemented


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_flatten(caa: ak.Array, daa: dak.Array, axis: int | None) -> None:
    cr = ak.flatten(caa.points.x, axis=axis)
    dr = dak.flatten(daa.points.x, axis=axis)
    assert_eq(cr, dr)


@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_num(caa: ak.Array, daa: dak.Array, axis: int) -> None:
    da = daa["points"]
    ca = caa["points"]

    if axis == 0:
        assert_eq(dak.num(da.x, axis=axis), ak.num(ca.x, axis=axis))
        da.eager_compute_divisions()
    if axis == 1:
        c1 = dak.num(da.x, axis=axis) > 2
        c2 = ak.num(ca.x, axis=axis) > 2
        assert_eq(da[c1], ca[c2])

    assert_eq(dak.num(da.x, axis=axis), ak.num(ca.x, axis=axis))


def test_zip_dict_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa["points"]["x"]
    da2 = daa["points"]["x"]
    ca1 = caa["points"]["x"]
    ca2 = caa["points"]["x"]

    da_z = dak.zip({"a": da1, "b": da2})
    ca_z = ak.zip({"a": ca1, "b": ca2})
    assert_eq(da_z, ca_z)


def test_unzip_dict_input(caa: ak.Array, daa: dak.Array) -> None:
    assert_eq(dak.zip(dak.unzip(daa["points"])), ak.zip(ak.unzip(caa["points"])))


def test_unzip_tuple_record() -> None:
    array = ak.Array([(1, 1.1, "one"), (2, 2.2, "two")])
    dakarray = dak.from_awkward(array, 1)

    assert_eq(dak.zip(dak.unzip(dakarray)), ak.zip(ak.unzip(array)))


def test_unzip_not_record() -> None:
    array = ak.Array([[1, 2, 3], [4, 5], [6]])
    dakarray = dak.from_awkward(array, 1)

    assert_eq(dak.zip(dak.unzip(dakarray)), ak.zip(ak.unzip(array)))


def test_zip_list_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa.points.x
    ca1 = caa.points.x
    dz1 = dak.zip([da1, da1])
    cz1 = ak.zip([ca1, ca1])
    assert_eq(dz1, cz1)
    dz2 = dak.zip([da1, da1, da1])
    cz2 = ak.zip([ca1, ca1, ca1])
    assert_eq(dz2, cz2)


def test_zip_tuple_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa.points.x
    ca1 = caa.points.x
    dz1 = dak.zip((da1, da1))
    cz1 = ak.zip((ca1, ca1))
    assert_eq(dz1, cz1)
    dz2 = dak.zip((da1, da1, da1))
    cz2 = ak.zip((ca1, ca1, ca1))
    assert_eq(dz2, cz2)


def test_zip_bad_input(daa: dak.Array) -> None:
    da1 = daa.points.x
    gd = (x for x in (da1, da1))
    with pytest.raises(
        DaskAwkwardNotImplemented, match="only mappings or sequences are supported"
    ):
        dak.zip(gd)


def test_zip_no_broadcast_dict_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa["points"]["x"]
    da2 = daa["points"]["x"]
    ca1 = caa["points"]["x"]
    ca2 = caa["points"]["x"]

    da_z = dak.zip_no_broadcast({"a": da1, "b": da2})
    ca_z = ak.zip_no_broadcast({"a": ca1, "b": ca2})
    assert_eq(da_z, ca_z)


def test_zip_no_broadcast_list_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa.points.x
    ca1 = caa.points.x
    dz1 = dak.zip_no_broadcast([da1, da1])
    cz1 = ak.zip_no_broadcast([ca1, ca1])
    assert_eq(dz1, cz1)
    dz2 = dak.zip_no_broadcast([da1, da1, da1])
    cz2 = ak.zip_no_broadcast([ca1, ca1, ca1])
    assert_eq(dz2, cz2)


def test_zip_no_broadcast_tuple_input(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa.points.x
    ca1 = caa.points.x
    dz1 = dak.zip_no_broadcast((da1, da1))
    cz1 = ak.zip_no_broadcast((ca1, ca1))
    assert_eq(dz1, cz1)
    dz2 = dak.zip_no_broadcast((da1, da1, da1))
    cz2 = ak.zip_no_broadcast((ca1, ca1, ca1))
    assert_eq(dz2, cz2)


def test_zip_no_broadcast_bad_input(daa: dak.Array) -> None:
    da1 = daa.points.x
    gd = (x for x in (da1, da1))
    with pytest.raises(
        DaskAwkwardNotImplemented, match="only mappings or sequences are supported"
    ):
        dak.zip_no_broadcast(gd)


def test_cartesian(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa["points", "x"]
    da2 = daa["points", "y"]
    ca1 = caa["points", "x"]
    ca2 = caa["points", "y"]

    dz = dak.cartesian([da1, da2], axis=1)
    cz = ak.cartesian([ca1, ca2], axis=1)
    assert_eq(dz, cz)

    dz = dak.cartesian({"xx": da1, "yy": da2}, axis=1)
    cz = ak.cartesian({"xx": ca1, "yy": ca2}, axis=1)
    assert_eq(dz, cz)


def test_argcartesian(caa: ak.Array, daa: dak.Array) -> None:
    da1 = daa["points", "x"]
    da2 = daa["points", "y"]
    ca1 = caa["points", "x"]
    ca2 = caa["points", "y"]

    dz = dak.argcartesian([da1, da2], axis=1)
    cz = ak.argcartesian([ca1, ca2], axis=1)
    assert_eq(dz, cz)

    dz = dak.argcartesian({"xx": da1, "yy": da2}, axis=1)
    cz = ak.argcartesian({"xx": ca1, "yy": ca2}, axis=1)
    assert_eq(dz, cz)


def test_ones_like(caa: ak.Array, daa: dak.Array) -> None:
    da1 = dak.ones_like(daa.points.x)
    ca1 = ak.ones_like(caa["points", "x"])
    assert_eq(da1, ca1)


def test_zeros_like(caa: ak.Array, daa: dak.Array) -> None:
    da1 = dak.zeros_like(daa["points", "x"])
    ca1 = ak.zeros_like(caa.points.x)
    assert_eq(da1, ca1)


@pytest.mark.parametrize("vf", [9, 99.9])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_fill_none(vf: int | float | str, axis: int | None) -> None:
    a = [[1, 2, None], [], [None], [5, 6, 7, None], [1, 2], None]
    b = [[None, 2, 1], [None], [], None, [7, 6, None, 5], [None, None]]
    c = dak.from_lists([a, b])
    d = dak.fill_none(c, vf, axis=axis)
    e = ak.fill_none(ak.from_iter(a + b), vf, axis=axis)
    assert_eq(d, e, check_forms=(not isinstance(vf, str)))


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_drop_none(axis: int) -> None:
    a = [[1, 2, None], [], [None], [5, 6, 7, None], [1, 2], None]
    b = [[None, 2, 1], [None], [], None, [7, 6, None, 5], [None, None]]
    c = dak.from_lists([a, b])
    d = dak.drop_none(c)
    e = ak.drop_none(ak.from_iter(a + b))
    assert_eq(d, e)


def test_nan_to_num():
    a = [[1, 2, np.nan], [], [np.nan], [5, 6, 7, np.nan], [1, 2], np.nan]
    b = [[np.nan, 2, 1], [np.nan], [], np.nan, [7, 6, np.nan, 5], [np.nan, np.nan]]
    c = dak.from_lists([a, b])
    d = dak.nan_to_num(c, nan=5)
    e = ak.nan_to_num(ak.from_iter(a + b), nan=5)
    assert_eq(d, e)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_is_none(axis: int) -> None:
    a: list[Any] = [[1, 2, None], None, None, [], [None], [5, 6, 7, None], [1, 2], None]
    b: list[Any] = [[None, 2, 1], [None], [], None, [7, 6, None, 5], [None, None]]
    c = dak.from_lists([a, b])
    d = dak.is_none(c, axis=axis)
    e = ak.is_none(ak.from_iter(a + b), axis=axis)
    assert_eq(d, e)


def test_local_index(daa, caa):
    assert_eq(
        dak.local_index(daa, axis=1),
        ak.local_index(caa, axis=1),
    )


def test_mask(daa, caa):
    mask = ak.any(caa.points.x > 3, axis=1)
    dmask = dak.any(daa.points.x > 3, axis=1)

    assert_eq(daa.mask[dmask], caa.mask[mask])
    assert_eq(dak.mask(daa, dmask), ak.mask(caa, mask))


def test_mask_from_awkward(daa, caa):
    mask = ak.any(caa.points.x > 3, axis=1)
    dmask = dak.from_awkward(mask, daa.npartitions)

    assert_eq(dak.mask(daa, dmask), ak.mask(caa, mask))


@pytest.mark.parametrize("axis", [1, -1, 2, -2])
@pytest.mark.parametrize("target", [5, 10, 1])
def test_pad_none(axis: int, target: int) -> None:
    a = [[1, 2, 3], [4], None]
    b = [[7], [], None, [6, 7, 8]]
    c = dak.from_lists([[a, b], [b, a]])
    d = ak.from_iter([a, b] + [b, a])
    assert_eq(
        dak.pad_none(c, target=target, axis=axis),
        ak.pad_none(d, target=target, axis=axis),
    )


def test_with_field(caa: ak.Array, daa: dak.Array) -> None:
    new_caa = ak.with_field(caa["points"], caa["points"]["x"], where="xx")
    new_daa = dak.with_field(daa["points"], daa["points"]["x"], where="xx")
    assert_eq(new_caa, new_daa)
    assert_eq(ak.without_field(new_caa, "xx"), ak.without_field(new_daa, "xx"))

    new_caa = ak.with_field(caa["points"], 1, where="xx")
    new_daa = dak.with_field(daa["points"], 1, where="xx")
    assert_eq(new_caa, new_daa)
    assert_eq(ak.without_field(new_caa, "xx"), ak.without_field(new_daa, "xx"))

    new_caa = ak.with_field(caa["points"], 1.0, where="xx")
    new_daa = dak.with_field(daa["points"], 1.0, where="xx")
    assert_eq(new_caa, new_daa)
    assert_eq(ak.without_field(new_caa, "xx"), ak.without_field(new_daa, "xx"))

    with pytest.raises(
        ValueError,
        match="Base argument in with_field must be a dask_awkward.Array",
    ):
        _ = dak.with_field([{"foo": 1.0}, {"foo": 2.0}], daa.points.x, where="x")

    with pytest.raises(
        ValueError,
        match="Base argument in without_field must be a dask_awkward.Array",
    ):
        _ = dak.without_field(
            [{"foo": [1.0, 2.0], "bar": [3.0, 4.0]}],
            "bar",
        )

    with pytest.raises(
        ValueError,
        match="with_field cannot accept string, bytes, list, or dict values yet",
    ):
        _ = dak.with_field(daa["points"], "hi there", where="q")


def test_setitem(caa: ak.Array, daa: dak.Array) -> None:
    daa["xx"] = daa["points"]["x"]
    caa["xx"] = caa["points"]["x"]

    daa["points", "z"] = np.sqrt(daa.points.x**2 + daa.points.y**2)
    caa["points", "z"] = np.sqrt(caa.points.x**2 + caa.points.y**2)
    assert_eq(caa, daa)

    with pytest.raises(
        DaskAwkwardNotImplemented,
        match="Supplying anything other than a dak.Array, or Number to __setitem__ is not yet available!\n\nIf you would like this unsupported call to be supported by\ndask-awkward please open an issue at:\nhttps://github.com/dask-contrib/dask-awkward.",
    ):
        daa["points", "q"] = "hi there"


def test_with_parameter() -> None:
    a = [[1, 2, 3], [], [4]]
    b = [[], [3], []]
    c = dak.from_lists([a, b])
    d = dak.with_parameter(c, "something", {})
    x = ak.from_iter(a + b)
    y = ak.with_parameter(x, "something", {})
    assert_eq(d, y)

    assert d.compute().layout.parameters == y.layout.parameters
    assert d._meta.layout.parameters == y.layout.parameters

    d2 = dak.without_parameters(d)
    y2 = ak.without_parameters(y)
    assert_eq(d2, y2)
    assert not d2.compute().layout.parameters
    assert d2.compute().layout.parameters == y2.layout.parameters
    assert d2._meta.layout.parameters == y2.layout.parameters


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("fields", [None, ["a", "b"]])
def test_combinations(caa, daa, axis, fields):
    assert_eq(
        dak.combinations(daa, 2, axis=axis),
        ak.combinations(caa, 2, axis=axis),
    )


def test_combinations_raise(daa):
    with pytest.raises(ValueError, match="if provided, the length"):
        dak.combinations(daa, 2, fields=["a", "b", "c"])


@pytest.mark.parametrize("axis", [1, -1])
@pytest.mark.parametrize("fields", [None, ["a", "b"]])
def test_argcombinations(caa, daa, axis, fields):
    if axis < 0:
        with pytest.raises(
            ValueError, match="the 'axis' for argcombinations must be non-negative"
        ):
            dak.argcombinations(daa, 2, axis=axis)
    else:
        assert_eq(
            dak.argcombinations(daa, 2, axis=axis),
            ak.argcombinations(caa, 2, axis=axis),
        )


def test_argcombinations_raise(daa):
    with pytest.raises(ValueError, match="if provided, the length"):
        dak.argcombinations(daa, 2, fields=["a", "b", "c"])


@pytest.mark.parametrize("mergebool", [True, False])
def test_where(caa, daa, mergebool):
    assert_eq(
        dak.where(
            daa.points.x > daa.points.y, daa.points.x, daa.points.y, mergebool=mergebool
        ),
        ak.where(
            caa.points.x > caa.points.y, caa.points.x, caa.points.y, mergebool=mergebool
        ),
    )

    assert_eq(
        dak.where(
            daa.points.x > daa.points.y, daa.points.x, 9999.0, mergebool=mergebool
        ),
        ak.where(
            caa.points.x > caa.points.y, caa.points.x, 9999.0, mergebool=mergebool
        ),
    )

    assert_eq(
        dak.where(
            daa.points.x > daa.points.y, 9999.0, daa.points.y, mergebool=mergebool
        ),
        ak.where(
            caa.points.x > caa.points.y, 9999.0, caa.points.y, mergebool=mergebool
        ),
    )

    assert_eq(
        dak.where(daa.points.x > daa.points.y, 8888.0, 9999.0, mergebool=mergebool),
        ak.where(caa.points.x > caa.points.y, 8888.0, 9999.0, mergebool=mergebool),
    )


def test_isclose(daa, caa):
    assert_eq(
        dak.isclose(daa.points.x, daa.points.y),
        ak.isclose(caa.points.x, caa.points.y),
    )


def test_singletons(daa, L4, tmp_path):
    pytest.importorskip("pyarrow")

    import warnings

    path = str(tmp_path)

    warnings.simplefilter("error")
    caa_L4 = ak.Array(L4)
    daa_L4 = dak.from_awkward(caa_L4, 1)
    assert_eq(
        dak.singletons(daa_L4),
        ak.singletons(caa_L4),
    )

    dak.to_parquet(daa, path)

    fpq_daa = dak.from_parquet(path)
    fpq_caa = ak.from_parquet(path)

    temp_zip = dak.zip({"x": fpq_daa.points.x, "y": fpq_daa.points.y})

    argmin_check = dak.singletons(dak.argmin(temp_zip.x, axis=1))

    assert_eq(
        argmin_check,
        ak.singletons(
            ak.argmin(ak.zip({"x": fpq_caa.points.x, "y": fpq_caa.points.y}).x, axis=1)
        ),
    )


@pytest.mark.parametrize("ascending", [True, False])
def test_argsort(daa, caa, ascending):
    assert_eq(
        dak.argsort(daa.points.x, ascending=ascending),
        ak.argsort(caa.points.x, ascending=ascending),
    )

    x = [[[1, 2, 3], [5, 4]], [[3, 1], [2, 3]]]
    a = dak.from_lists(x)
    b = ak.concatenate([ak.Array(x[0]), ak.Array(x[1])])
    assert_eq(
        a[dak.argsort(a, axis=1, ascending=ascending)],
        b[ak.argsort(b, axis=1, ascending=ascending)],
    )


@pytest.mark.parametrize("ascending", [True, False])
def test_sort(daa, caa, ascending):
    assert_eq(
        dak.sort(daa.points.x, ascending=ascending),
        ak.sort(caa.points.x, ascending=ascending),
    )


def test_copy(daa):
    result = dak.copy(daa)
    assert result._meta is not daa._meta


@pytest.mark.parametrize(
    "thedtype",
    [
        bool,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
        pytest.param(
            np.datetime64,
            marks=pytest.mark.xfail(AK_LTE_2_5_0, reason="np dtype problem"),
        ),
        np.timedelta64,
        np.float16,
    ],
)
def test_full_like(daa, caa, thedtype):
    value = 12.6
    if thedtype is np.datetime64:
        value = thedtype(int(value), "us")
        thedtype = np.dtype("datetime64[us]")
    elif thedtype is np.timedelta64:
        value = thedtype(int(value))

    assert_eq(
        dak.full_like(daa, value, dtype=thedtype),
        ak.full_like(caa, value, dtype=thedtype),
    )


def test_unflatten(daa, caa):
    counts = ak.Array([2, 3, 0, 5, 3, 2])
    dcounts = dak.from_awkward(counts, daa.npartitions)

    assert_eq(
        dak.unflatten(daa, dcounts),
        ak.unflatten(caa, counts),
    )


def test_to_packed(daa, caa):
    assert_eq(
        dak.to_packed(daa),
        ak.to_packed(caa),
    )


def test_to_list(daa):
    assert dak.to_list(daa) == daa.compute().to_list()


def test_ravel(daa, caa):
    assert_eq(
        dak.ravel(daa.points.x),
        ak.ravel(caa.points.x),
    )


@pytest.mark.xfail
def test_ravel_fail(daa, caa):
    assert_eq(
        dak.ravel(daa),
        ak.ravel(caa),
    )


def test_run_lengths(daa, caa):
    assert_eq(
        dak.run_lengths(daa.points.x),
        ak.run_lengths(caa.points.x),
    )


def test_from_regular(caa):
    regular = ak.to_regular(ak.to_packed(caa[[0, 4, 5, 9, 10, 14]].points.x))
    dregular = dak.from_awkward(regular, 3)

    assert_eq(
        dak.from_regular(dregular, axis=1),
        ak.from_regular(regular, axis=1),
    )


def test_to_regular(caa):
    regular = ak.to_packed(caa[[0, 4, 5, 9, 10, 14]].points.x)
    dregular = dak.from_awkward(regular, 3)

    assert_eq(
        dak.to_regular(dregular, axis=1),
        ak.to_regular(regular, axis=1),
    )


def test_broadcast_arrays(daa, caa):
    flat = ak.Array([1] * 15)
    dflat = dak.from_awkward(flat, 3)

    dak_broadcast = dak.broadcast_arrays(dflat, daa.points.x)
    ak_broadcast = ak.broadcast_arrays(flat, caa.points.x)

    assert len(dak_broadcast) == len(ak_broadcast)

    for db, b in zip(dak_broadcast, ak_broadcast):
        assert_eq(db, b)


def test_values_astype(daa, caa):
    assert_eq(
        dak.values_astype(daa, np.float32),
        ak.values_astype(caa, np.float32),
    )


def test_repartition_whole(daa):
    daa1 = daa.repartition(npartitions=1)
    assert daa1.npartitions == 1
    assert_eq(daa, daa1, check_divisions=False)


def test_repartition_one_to_n(daa):
    daa1 = daa.repartition(one_to_n=2)
    assert daa1.npartitions == daa.npartitions * 2
    assert_eq(daa, daa1, check_divisions=False)


def test_repartition_n_to_one():
    daa = dak.from_lists([[[1, 2, 3], [], [4, 5]] * 2] * 52)
    daa2 = daa.repartition(n_to_one=52)
    assert daa2.npartitions == 1
    assert daa.compute().to_list() == daa2.compute().to_list()
    daa2 = daa.repartition(n_to_one=53)
    assert daa2.npartitions == 1
    assert daa.compute().to_list() == daa2.compute().to_list()
    daa2 = daa.repartition(n_to_one=2)
    assert daa2.npartitions == 26
    assert daa.compute().to_list() == daa2.compute().to_list()
    daa2 = daa.repartition(n_to_one=10)
    assert daa2.npartitions == 6
    assert daa.compute().to_list() == daa2.compute().to_list()
    daa._divisions = (None,) * len(daa.divisions)
    assert daa2.npartitions == 6
    assert daa.compute().to_list() == daa2.compute().to_list()


def test_repartition_no_change(daa):
    daa1 = daa.repartition(divisions=(0, 5, 10, 15))
    assert daa1.npartitions == 3
    assert_eq(daa, daa1, check_divisions=False)


def test_repartition_split_all(daa):
    daa1 = daa.repartition(rows_per_partition=1)
    assert daa1.npartitions == len(daa)
    out = daa1.compute()
    assert out.tolist() == daa.compute().tolist()


def test_repartition_uneven(daa):
    daa1 = daa.repartition(divisions=(0, 7, 8, 11, 12))
    assert daa1.npartitions == 4
    out = daa1.compute()
    assert out.tolist() == daa.compute()[:12].tolist()
