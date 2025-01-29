from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import awkward as ak
from awkward.operations.ak_concatenate import (
    enforce_concatenated_form as enforce_layout_to_concatenated_form,
)
from awkward.typetracer import typetracer_from_form
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from dask_awkward.layers import AwkwardMaterializedLayer
from dask_awkward.lib.core import (
    Array,
    PartitionCompatibility,
    map_partitions,
    new_array_object,
    partition_compatibility,
)
from dask_awkward.utils import DaskAwkwardNotImplemented, IncompatiblePartitions

if TYPE_CHECKING:
    from awkward.forms import Form
    from awkward.highlevel import Array as AwkwardArray


class ConcatenateFnAxisGT0:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args):
        return ak.concatenate(list(args), **self.kwargs)


def _enforce_concatenated_form(array: AwkwardArray, form: Form) -> AwkwardArray:
    layout = ak.to_layout(array)
    # TODO: should this check whether the form agrees first, or assume that the
    #       operation is harmless if not required?
    result = enforce_layout_to_concatenated_form(layout, form)
    return ak.Array(result, behavior=array._behavior, attrs=array._attrs)


from awkward.typetracer import TypeTracerReport


class ParentReport(TypeTracerReport):
    def __init__(self):
        self._parent_to_child: dict[str, tuple[TypeTracerReport, str]] = {}

    def add_child_key(
        self, parent_key: str, child_key: str, child_report: TypeTracerReport
    ):
        self._parent_to_child.setdefault(parent_key, []).append(
            (child_report, child_key)
        )

    @property
    def shape_touched(self):
        raise NotImplementedError

    @property
    def data_touched(self):
        raise NotImplementedError

    def touch_shape(self, label: str):
        if (child_infos := self._parent_to_child.get(label)) is not None:
            for child_report, child_label in child_infos:
                child_report.touch_shape(child_label)

    def touch_data(self, label: str):
        if (child_infos := self._parent_to_child.get(label)) is not None:
            for child_report, child_label in child_infos:
                child_report.touch_data(child_label)


def maybe_parent_report(parent, children, parent_report):
    if parent_report is None:
        parent_report = ParentReport()
    if parent.report is not None:
        parent_report.add_child_key(parent.form_key, parent.form_key, parent.report)
    for child in children:
        if child.report is not None:
            parent_report.add_child_key(parent.form_key, child.form_key, child.report)
    parent.report = parent_report
    return parent_report


def merge_reports(first, *remainder):
    parent_report = None

    def impl(first, *remainder):
        nonlocal parent_report
        assert all(type(rem) is type(first) for rem in remainder)

        if first.is_numpy:
            parent_report = maybe_parent_report(
                first.data, [c.data for c in remainder], parent_report
            )

        elif first.is_option and first.is_indexed:
            parent_report = maybe_parent_report(
                first.index.data, [c.index.data for c in remainder], parent_report
            )
            impl(first.content, *[c.content for c in remainder])

        elif first.is_option:
            parent_report = maybe_parent_report(
                first.mask.data, [c.mask.data for c in remainder], parent_report
            )
            impl(first.content, *[c.content for c in remainder])

        elif first.is_list and isinstance(first, ak.contents.ListOffsetArray):
            parent_report = maybe_parent_report(
                first.offsets.data, [c.offsets.data for c in remainder], parent_report
            )
            impl(first.content, *[c.content for c in remainder])

        elif first.is_list and isinstance(first, ak.contents.ListArray):
            parent_report = maybe_parent_report(
                first.starts.data, [c.starts.data for c in remainder], parent_report
            )
            parent_report = maybe_parent_report(
                first.stops.data, [c.stops.data for c in remainder], parent_report
            )
            impl(first.content, *[c.content for c in remainder])

        elif first.is_list and isinstance(first, ak.contents.RegularArray):
            impl(first.content, *[c.content for c in remainder])

        elif first.is_indexed:
            parent_report = maybe_parent_report(
                first.index.data, [c.index.data for c in remainder], parent_report
            )
            impl(first.content, *[c.content for c in remainder])

        elif first.is_record:
            for this, *that in zip(first.contents, *[c.contents for c in remainder]):
                impl(this, *that)

        elif first.is_empty:
            return

        elif first.is_union:
            raise NotImplementedError

        else:
            raise AssertionError

    impl(first, *remainder)


def _concatenate_axis_0_meta(*arrays: AwkwardArray) -> AwkwardArray:
    # At this stage, the metas have all been enforced to the same type
    layouts = [arr.layout for arr in arrays]
    merge_reports(layouts[0], *layouts)
    return arrays[0]


def concatenate(
    arrays: list[Array],
    axis: int = 0,
    mergebool: bool = True,
    highlevel: bool = True,
    behavior: Mapping | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> Array:
    label = "concatenate"
    token = tokenize(arrays, axis, mergebool, highlevel, behavior)
    name = f"{label}-{token}"

    metas = [c._meta for c in arrays]

    if len(metas) == 0:
        raise ValueError("Need at least one array to concatenate")

    # Are we performing a _logical_ concatenation?
    if axis == 0:
        # There are two possible cases here:
        # 1. all arrays have identical metas — just grow the Dask collection
        # 2. some arrays have different metas — coerce arrays to same form

        # Drop reports from metas to avoid later touching any buffers
        metas_no_report = [
            typetracer_from_form(x.layout.form, behavior=x.behavior, attrs=x._attrs)
            for x in metas
        ]
        # Concatenate metas to determine result form
        meta_no_report = ak.concatenate(
            metas_no_report, axis=0, behavior=behavior, attrs=attrs
        )
        intended_form = meta_no_report.layout.form

        # If any forms aren't equal to this form, we must enforce each form to the same type
        if any(
            not m.layout.form.is_equal_to(
                intended_form, all_parameters=True, form_key=False
            )
            for m in metas
        ):
            arrays = [
                map_partitions(
                    _enforce_concatenated_form,
                    c,
                    label="enforce-concat-form",
                    form=intended_form,
                    output_divisions=1,
                )
                for c in arrays
            ]

            g = {
                (name, i): k
                for i, k in enumerate(
                    k for collection in arrays for k in collection.__dask_keys__()
                )
            }

            aml = AwkwardMaterializedLayer(
                g,
                previous_layer_names=[a.name for a in arrays],
                fn=_concatenate_axis_0_meta,
            )
        else:
            g = {
                (name, i): k
                for i, k in enumerate(
                    k for collection in arrays for k in collection.__dask_keys__()
                )
            }

            aml = AwkwardMaterializedLayer(
                g,
                previous_layer_names=[a.name for a in arrays],
                fn=_concatenate_axis_0_meta,
            )

        hlg = HighLevelGraph.from_collections(name, aml, dependencies=arrays)
        return new_array_object(
            hlg,
            name,
            meta=meta_no_report,
            npartitions=sum(a.npartitions for a in arrays),
        )

    if axis > 0:
        if partition_compatibility(*arrays) == PartitionCompatibility.NO:
            raise IncompatiblePartitions("concatenate", *arrays)

        fn = ConcatenateFnAxisGT0(axis=axis, behavior=behavior, attrs=attrs)
        return map_partitions(fn, *arrays, label="concatenate-axisgt0")

    raise DaskAwkwardNotImplemented("TODO")
