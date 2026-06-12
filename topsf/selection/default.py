# coding: utf-8

"""
Exemplary selection methods.
"""
from __future__ import annotations
import law

from operator import and_
from functools import reduce
from collections import defaultdict

from columnflow.util import maybe_import, DotDict
from columnflow.columnar_util import remove_ak_column, EMPTY_FLOAT

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.selection.cms.btag import fill_btag_wp_count_hists

from topsf.selection.lepton import lepton_selection
from topsf.selection.jet import jet_selection, jet_lepton_2d_selection
from topsf.selection.bjet import bjet_lepton_selection
from topsf.selection.fatjet import fatjet_selection
from topsf.selection.met import met_selection
from topsf.selection.w_lep import w_lep_selection
from topsf.selection.cutflow_features import cutflow_features
from topsf.selection.common import get_weights_and_no_sel_mask, pre_selection
from topsf.selection.stats import topsf_increment_stats, topsf_selection_step_stats
from topsf.selection.hists import topsf_selection_hists

from topsf.production.processes import process_ids
from topsf.production.probe_jet import probe_jet
from topsf.production.gen_top import gen_parton_top
from topsf.production.gen_v import gen_v_boson

from topsf.util import has_tag, record_calls

from columnflow.production.categories import category_ids
# from topsf.production.categories import category_ids


np = maybe_import("numpy")
ak = maybe_import("awkward")
hist = maybe_import("hist")

logger = law.logger.get_logger(__name__)


@selector(
    uses={
        pre_selection,
        process_ids, category_ids,
        cutflow_features,
        lepton_selection,
        met_selection,
        w_lep_selection,
        jet_selection,
        bjet_lepton_selection,
        jet_lepton_2d_selection,
        fatjet_selection,
        probe_jet,
        gen_parton_top,
        gen_v_boson,
        get_weights_and_no_sel_mask,
        topsf_selection_step_stats,
        topsf_increment_stats,
        topsf_selection_hists,
    },
    produces={
        pre_selection,
        process_ids, category_ids,
        cutflow_features,
        lepton_selection,
        met_selection,
        w_lep_selection,
        jet_selection,
        bjet_lepton_selection,
        jet_lepton_2d_selection,
        fatjet_selection,
        probe_jet,
        gen_parton_top,
        gen_v_boson,
        get_weights_and_no_sel_mask,
        topsf_selection_step_stats,
        topsf_increment_stats,
        topsf_selection_hists,
    },
    exposed=True,
)
def default(
    self: Selector,
    events: ak.Array,
    stats: defaultdict,
    hists: DotDict[str, hist.Hist],
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    run_list = []
    with record_calls(self, run_list):
        # ensure coffea behavior
        events, results = self[pre_selection](events, stats, **kwargs)

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

        # derive event weights and add base mask of all events that are not considered bad to "cleanup" step
        events, results = self[get_weights_and_no_sel_mask](events, results, **kwargs)
        results.steps["cleanup"] = results.steps.cleanup & results.steps["no_sel_mask"]

        results.steps["all_but_trigger_and_bjet"] = (
            results.steps.cleanup &
            results.steps.Lepton &
            results.steps.AddleptonVeto &
            results.steps.Jet &
            results.steps.JetLepton2DCut &
            results.steps.FatJet &
            results.steps.MET &
            results.steps.WLepPt
        )

        results.steps["all_but_bjet"] = (
            results.steps.cleanup &
            results.steps.LeptonTrigger &
            results.steps.Lepton &
            results.steps.AddleptonVeto &
            results.steps.Jet &
            results.steps.JetLepton2DCut &
            results.steps.FatJet &
            results.steps.MET &
            results.steps.WLepPt
        )

        results.steps["all"] = (
            results.steps.all_but_bjet &
            results.steps.BJetLeptonDeltaR
        )

        # combined event selection after all steps
        event_sel = reduce(and_, results.steps.values())
        results.event = event_sel

        for step, sel in results.steps.items():
            n_sel = ak.sum(sel, axis=-1)
            logger.debug(f"{step}: {n_sel}")

        n_sel = ak.sum(event_sel, axis=-1)
        if n_sel - ak.sum(results.steps['all']) != 0:
            logger.debug(f"__all__: {n_sel}")
            logger.warning_once(
                f"Number of events passing combined selection does not match number of events passing all individual steps: {n_sel} vs {ak.sum(results.steps['all'])}"  # noqa
            )
            raise ValueError("Inconsistent event selection results")

        # produce features relevant for selection and event weights
        if self.dataset_inst.has_tag("is_ttbar"):
            events = self[gen_parton_top](events, **kwargs)

        if self.dataset_inst.has_tag("is_v_jets"):
            events = self[gen_v_boson](events, **kwargs)

        events = self[probe_jet](events, **kwargs)

        # create process ids
        events = self[process_ids](events, **kwargs)

        # build categories
        events = self[category_ids](events, results=results, **kwargs)

        # add cutflow features
        events = self[cutflow_features](events, object_masks=results.objects, **kwargs)

        # increment stats
        events = self[topsf_selection_step_stats](events, results, stats, **kwargs)
        events = self[topsf_increment_stats](events, results, stats, **kwargs)
        events = self[topsf_selection_hists](events, results, hists, **kwargs)

        if self.dataset_inst.is_mc and has_tag("skip_btag_weights", self.config_inst, self.dataset_inst, operator=any):
            self[fill_btag_wp_count_hists](events, results.event, results.objects.Jet.Jet, hists, **kwargs)

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

        # remove unused columns
        for col in ["GenPart", "GenPartonTop"]:
            for field in [
                "genPartIdxMother",
                "statusFlags",
                "genPartIdxMotherG",
                "distinctParentIdxG",
                "childrenIdxG",
                "distinctChildrenIdxG",
                "distinctChildrenDeepIdxG",
            ]:
                events = remove_ak_column(events, f"{col}.{field}", silent=True)

        # avoid none values in events
        events = ak.fill_none(events, EMPTY_FLOAT)

        logger.info(f"Selected {ak.sum(results.event)} from {len(events)} events")

    logger.info_once(
        "Finished default selector steps:\n" +
        "\n".join(run_list)
    )

    return events, results


@default.init
def default_init(self: Selector):
    # Add shift dependencies
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }

    if hasattr(self, "dataset_inst") and self.dataset_inst.is_mc:
        self.uses |= {
            fill_btag_wp_count_hists,
        }
        self.produces |= {
            fill_btag_wp_count_hists,
        }
