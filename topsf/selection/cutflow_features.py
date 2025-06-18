# coding: utf-8

"""
Selectors to set ak columns for cutflow features
"""

from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column, Route, EMPTY_FLOAT
from columnflow.production import Producer, producer

from columnflow.production.cms.mc_weight import mc_weight

from topsf.production.categories import category_ids

ak = maybe_import("awkward")


@producer(
    uses={
        # needed for cf.PlotCutflow* tasks
        mc_weight, category_ids,
        # nano columns
        "Jet.pt", "Jet.eta", "Jet.phi",
    },
    produces={
        # needed for cf.PlotCutflow* tasks
        mc_weight, category_ids,
        # new columns
        "cutflow.jet1_pt",
    },
)
def cutflow_features(
    self: Producer,
    events: ak.Array,
    object_masks: dict[str, dict[str, ak.Array]],
    **kwargs,
) -> ak.Array:
    # apply weights in MC
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # apply object selections
    selected_jet = events.Jet[object_masks["Jet"]["Jet"]]

    events = set_ak_column(events, "cutflow.jet1_pt", Route("pt[:,0]").apply(selected_jet, EMPTY_FLOAT))

    return events
