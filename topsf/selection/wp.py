# coding: utf-8

"""
Selection methods related to WP analysis.
"""
from __future__ import annotations
import law

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT

from columnflow.selection import Selector, SelectionResult, selector

# from columnflow.production.cms.jet import jet_id. # FIXME recalculate jetId in Nano version > v12
from columnflow.production.processes import process_ids

from topsf.selection.util import masked_sorted_indices
from topsf.selection.stats import topsf_increment_stats

from topsf.production.wp import wp_category_ids
from topsf.production.gen_top import gen_parton_top

from topsf.selection.common import get_weights_and_no_sel_mask, pre_selection
from topsf.util import has_tag

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


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
        (fatjet.pt > self.cfg.min_pt) &
        (fatjet.jetId & self.cfg.jetId == self.cfg.jetId)  # jetId bitmask
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
        f"{column}.jetId",
    }

    # if ttbar, produce parton-level top quarks
    # (relevant for jet selection)
    dataset_inst = getattr(self, "dataset_inst", None)
    if dataset_inst is not None and dataset_inst.has_tag("is_ttbar"):
        self.uses.add(gen_parton_top)
        self.produces.add(gen_parton_top)


@selector(
    uses={
        pre_selection,
        wp_category_ids,
        process_ids,
        wp_fatjet_selection,
        topsf_increment_stats,
        get_weights_and_no_sel_mask,
    },
    produces={
        pre_selection,
        wp_category_ids,
        process_ids,
        wp_fatjet_selection,
        topsf_increment_stats,
        get_weights_and_no_sel_mask,
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
    events, results = self[pre_selection](events, stats, **kwargs)

    # fatjet selection
    events, wp_fatjet_results = self[wp_fatjet_selection](
        events,
        msoftdrop_range=msoftdrop_range,
        **kwargs,
    )
    results += wp_fatjet_results

    # derive event weights and add base mask of all events that are not considered bad to "cleanup" step
    events, results = self[get_weights_and_no_sel_mask](events, results, **kwargs)
    results.steps["cleanup"] = results.steps.cleanup & results.steps["no_sel_mask"]

    results.steps["all"] = (
        results.steps.cleanup &
        results.steps.FatJet
    )

    # combined event selection after all steps
    event_sel = reduce(and_, results.steps.values())
    results.event = event_sel

    for step, sel in results.steps.items():
        n_sel = ak.sum(sel, axis=-1)
        logger.debug(f"{step}: {n_sel}")

    n_sel = ak.sum(event_sel, axis=-1)
    logger.debug(f"__all__: {n_sel}")

    # produce features relevant for selection and event weights
    if self.dataset_inst.has_tag("is_ttbar"):
        events = self[gen_parton_top](events, **kwargs)

    # build categories
    events = self[wp_category_ids](events, **kwargs)

    # create process ids
    events = self[process_ids](events, **kwargs)

    # increment stats
    events = self[topsf_increment_stats](events, results, stats, **kwargs)
    # no custom hists needed, because we don't do b tagging in wp analysis

    def log_fraction(stats_key: str, msg: str | None = None):
        if not stats.get(stats_key):
            return
        if not msg:
            msg = "Fraction of {stats_key}"
        logger.info(f"{msg}: {(100 * stats[stats_key] / stats['num_events']):.2f}%")

    log_fraction("num_negative_weights", "Fraction of negative weights")
    log_fraction("num_pu_0", "Fraction of events with pu_weight == 0")
    log_fraction("num_pu_100", "Fraction of events with pu_weight >= 100")

    # temporary fix for optional types from Calibration (e.g. events.Jet.pt --> ?float32)
    # TODO: remove as soon as possible as it might lead to weird bugs when there are none entries in inputs
    events = ak.fill_none(events, EMPTY_FLOAT)

    logger.info(f"Selected {ak.sum(results.event)} from {len(events)} events")

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
