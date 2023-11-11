# coding: utf-8

"""
Selection methods related to WP analysis.
"""

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.met_filters import met_filters

from columnflow.production.util import attach_coffea_behavior
from columnflow.production.processes import process_ids

from topsf.selection.util import masked_sorted_indices
from topsf.selection.default import increment_stats
from topsf.selection.fatjet import fatjet_selection

from topsf.production.wp import wp_category_ids
from topsf.production.gen_top import gen_parton_top
from topsf.production.gen_v import gen_v_boson


np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        "event",
    },
)
def wp_fatjet_selection(
    self: Selector,
    events: ak.Array,
    msoftdrop_range=None,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Select AK8 jets that are well separated from the lepton.
    """

    # get selection parameters from the config
    self.cfg = self.config_inst.x.jet_selection.get("ak8", "FatJet")

    # choose jet column
    fatjet = events[self.cfg.column]

    # select jets
    fatjet_mask = (
        (abs(fatjet.eta) < self.cfg.max_abseta) &
        (fatjet.pt > self.cfg.min_pt)
    )

    # resolve optional msoftdrop range
    msoftdrop_range = list(map(
        lambda x: float(x) if x is not None else x,
        msoftdrop_range or (None, None),
    ))
    if msoftdrop_range[0]:
        fatjet_mask = fatjet_mask & (fatjet.msoftdrop > msoftdrop_range[0])
    if msoftdrop_range[1]:
        fatjet_mask = fatjet_mask & (fatjet.msoftdrop < msoftdrop_range[1])

    # if ttbar sample, additionally check if fat jet is uniquely matched
    # to one of the parton-level top quarks
    if self.dataset_inst.has_tag("is_ttbar"):
        events = self[gen_parton_top](events, **kwargs)

        fatjet_delta_r_top = fatjet.metric_table(events.GenPartonTop)
        fatjet_n_matched_tops = ak.sum(fatjet_delta_r_top < 0.6, axis=-1)

        fatjet_mask = fatjet_mask & (fatjet_n_matched_tops == 1)

    # compute indices from mask
    fatjet_indices = masked_sorted_indices(fatjet_mask, fatjet.pt)

    # return selection result
    return events, SelectionResult(
        steps={
            "FatJet": ak.fill_none(ak.num(fatjet_indices) >= 1, False),
        },
        objects={
            self.cfg.column: {
                self.cfg.column: fatjet_indices,
            },
        },
    )


@wp_fatjet_selection.init
def wp_fatjet_selection_init(self: Selector) -> None:
    # return immediately if config not yet loaded
    config_inst = getattr(self, "config_inst", None)
    if not config_inst:
        return

    # set config dict
    self.cfg = config_inst.x.jet_selection.ak8

    # set input columns
    column = self.cfg.column
    self.uses |= {
        f"{column}.pt",
        f"{column}.eta",
        f"{column}.phi",
        f"{column}.mass",
        f"{column}.msoftdrop",
    }

    # if ttbar, produce parton-level top quarks
    # (relevant for jet selection)
    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst is not None and dataset_inst.has_tag("is_ttbar"):
        self.uses.add(gen_parton_top)


@selector(
    uses={
        attach_coffea_behavior,
        wp_category_ids,
        process_ids,
        met_filters,
        wp_fatjet_selection,
        increment_stats,
    },
    produces={
        wp_category_ids,
        process_ids,
        met_filters,
        wp_fatjet_selection,
        increment_stats,
    },
    exposed=True,
)
def wp(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    msoftdrop_range=None,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    # ensure coffea behavior
    events = self[attach_coffea_behavior](events, **kwargs)

    # prepare the selection results that are updated at every step
    results = SelectionResult()

    # MET filters
    events, met_filters_results = self[met_filters](events, **kwargs)
    results += met_filters_results

    # fatjet selection
    events, wp_fatjet_results = self[wp_fatjet_selection](
        events,
        msoftdrop_range=msoftdrop_range,
        **kwargs,
    )
    results += wp_fatjet_results

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.main["event"] = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        print(f"{step}: {n_sel}")

    n_sel = ak.sum(event_sel, axis=-1)
    print(f"__all__: {n_sel}")

    # produce features relevant for selection and event weights
    if self.dataset_inst.has_tag("is_ttbar"):
        events = self[gen_parton_top](events, **kwargs)

    # build categories
    events = self[wp_category_ids](events, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # increment stats
    self[increment_stats](events, results, stats, **kwargs)

    return events, results


@wp.init
def wp_init(self: Selector):
    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst is not None and dataset_inst.is_data:
        raise RuntimeError("selector 'wp' should not be run on data")

    # if ttbar, produce parton-level top quarks
    # (relevant for jet selection)
    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst and dataset_inst.has_tag("is_ttbar"):
        self.uses.add(gen_parton_top)
        self.produces.add(gen_parton_top)


# selector for msoftdrop region around top mass
msoftdrop_range = (105, 210)
@selector(
    uses={
        wp,
    },
    produces={
        wp,
    },
    cls_name=f"wp_msoftdrop_{'_'.join(map(str, msoftdrop_range))}",
    exposed=True,
)
def wp_msoftdrop_range(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    msoftdrop_range=msoftdrop_range,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:

    events, results = self[wp](events, stats, msoftdrop_range=msoftdrop_range, **kwargs)

    return events, results
