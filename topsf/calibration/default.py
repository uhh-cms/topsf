# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jets
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import

from topsf.calibration.jets import jet_lepton_cleaner

np = maybe_import("numpy")
ak = maybe_import("awkward")


@calibrator(
    uses={mc_weight, deterministic_seeds, jet_lepton_cleaner, jets},
    produces={mc_weight, deterministic_seeds, jet_lepton_cleaner, jets},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[jet_lepton_cleaner](events, **kwargs)  # set Jet.pt to raw Pt -> run before jets (JER) calibrator
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[jets](events, **kwargs)  # call_force ?
    events = self[deterministic_seeds](events, **kwargs)

    return events


@calibrator(
    uses={mc_weight, deterministic_seeds, jets},
    produces={mc_weight, deterministic_seeds, jets},
)
def no_jet_cleaning(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[jets](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)

    return events
