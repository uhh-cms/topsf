# coding: utf-8

"""
Exemplary selection methods.
"""

from operator import and_
from functools import reduce
from collections import defaultdict, OrderedDict

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters
from columnflow.selection.cms.json_filter import json_filter

from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.util import attach_coffea_behavior

from topsf.selection.lepton import lepton_selection
from topsf.selection.jet import jet_selection, jet_lepton_2d_selection
from topsf.selection.bjet import bjet_lepton_selection
from topsf.selection.fatjet import fatjet_selection
from topsf.selection.met import met_selection
from topsf.selection.w_lep import w_lep_selection
from topsf.selection.cutflow_features import cutflow_features

from topsf.production.processes import process_ids
from topsf.production.categories import category_ids
from topsf.production.probe_jet import probe_jet
from topsf.production.gen_top import gen_parton_top


np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={"mc_weight", "process_id"},
    check_columns_present={"produces"},  # some used columns optional
)
def increment_stats(
    self: Selector,
    events: ak.Array,
    results: SelectionResult,
    stats: dict,
    **kwargs,
) -> ak.Array:
    """
    Unexposed selector that does not actually select objects but instead increments selection
    *stats* in-place based on all input *events* and the final selection *mask*.
    """
    # get the event mask
    mask = results.main.event

    # ensure mask passed is boolean
    mask = ak.values_astype(mask, bool)

    # increment plain counts
    stats["n_events"] += len(events)
    stats["n_events_selected"] += ak.sum(mask, axis=0)

    # create a map of entry names to (weight, mask) pairs that will be written to stats
    weight_map = OrderedDict()
    if self.dataset_inst.is_mc:
        # mc weight for all events
        weight_map["mc_weight"] = (events.mc_weight, Ellipsis)

        # mc weight for selected events
        weight_map["mc_weight_selected"] = (events.mc_weight, mask)

        # add more entries here
        # ...

    # get and store the weights
    for name, (weights, mask) in weight_map.items():
        joinable_mask = True if mask is Ellipsis else mask

        # sum for all processes
        stats[f"sum_{name}"] += ak.sum(weights[mask])

        # sums per process id
        stats.setdefault(f"sum_{name}_per_process", defaultdict(float))
        processes = np.unique(events.process_id)
        for p in processes:
            stats[f"sum_{name}_per_process"][int(p)] += ak.sum(
                weights[(events.process_id == p) & joinable_mask],
            )

        # sums per category
        stats.setdefault(f"sum_{name}_per_category", defaultdict(float))
        categories = np.unique(ak.ravel(events.category_ids))
        for c in categories:
            stats[f"sum_{name}_per_category"][int(c)] += ak.sum(
                weights[ak.any(events.category_ids == c, axis=-1) & joinable_mask],
            )

    return events


@selector(
    uses={
        attach_coffea_behavior,
        mc_weight, process_ids, category_ids,
        cutflow_features,
        met_filters,
        lepton_selection,
        met_selection,
        w_lep_selection,
        jet_selection,
        bjet_lepton_selection,
        jet_lepton_2d_selection,
        fatjet_selection,
        increment_stats,
        probe_jet,
        gen_parton_top,
    },
    produces={
        mc_weight, process_ids, category_ids,
        cutflow_features,
        met_filters,
        lepton_selection,
        met_selection,
        w_lep_selection,
        jet_selection,
        bjet_lepton_selection,
        jet_lepton_2d_selection,
        fatjet_selection,
        increment_stats,
        probe_jet,
        gen_parton_top,
    },
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # MET filters
    results.steps.METFilters = self[met_filters](events, **kwargs)

    # JSON filter (data-only)
    if self.dataset_inst.is_data:
        results.steps.JSON = self[json_filter](events, **kwargs)

    # lepton selection
    events, lepton_results = self[lepton_selection](events, **kwargs)
    results += lepton_results

    # jet selection
    events, jet_results = self[jet_selection](events, **kwargs)
    results += jet_results

    # bjet-lepton selection
    events, bjet_lepton_results = self[bjet_lepton_selection](events, **kwargs)
    results += bjet_lepton_results

    # jet-lepton 2D selection
    events, jet_lepton_2d_results = self[jet_lepton_2d_selection](events, results=results, **kwargs)
    results += jet_lepton_2d_results

    # fatjet selection
    events, fatjet_results = self[fatjet_selection](events, **kwargs)
    results += fatjet_results

    # met selection
    events, met_results = self[met_selection](events, **kwargs)
    results += met_results

    # w_lep selection
    events, w_lep_results = self[w_lep_selection](events, **kwargs)
    results += w_lep_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        print(f"{step}: {n_sel}")

    n_sel = ak.sum(event_sel, axis=-1)
    print(f"__all__: {n_sel}")

    # produce features relevant for selection and event weights
    events = self[gen_parton_top](events, **kwargs)
    events = self[probe_jet](events, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # build categories
    events = self[category_ids](events, results=results, **kwargs)

    # add cutflow features
    events = self[cutflow_features](events, object_masks=results.objects, **kwargs)

    # increment stats
    self[increment_stats](events, results, stats, **kwargs)

    return events, results


@default.init
def default_init(self: Selector):
    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst is not None and dataset_inst.is_data:
        self.uses |= {json_filter}
