# coding: utf-8

"""
Calibration methods.
"""
import functools

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jets_ak4, jets_ak8
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.jet import msoftdrop
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from topsf.calibration.jets import jet_lepton_cleaner, jec_subjets, jer_subjets

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


@calibrator(
    uses={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        jets_ak4,
        jets_ak8,
        jec_subjets,
        # jer_subjets,
        msoftdrop,
    },
    produces={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        jets_ak4,
        jets_ak8,
        jec_subjets,
        # jer_subjets,
        msoftdrop,
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    events = self[jet_lepton_cleaner](events, **kwargs)  # set Jet.pt to raw Pt -> run before jets (JER) calibrator
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[jets_ak4](events, **kwargs)  # call_force ?
    events = self[jets_ak8](events, **kwargs)  # call_force ?

    events = self[deterministic_seeds](events, **kwargs)

    # fake subjet area column by setting it to an array with the same structure as the subjet pt column containing 0.5
    # (needed to be able to use same code as for top-level AK4/AK8 jets, as the producer formally requires an `area` column, despite not actually using it)
    events = set_ak_column_f32(events, "SubJet.area", 0.5 * ak.ones_like(events.SubJet.pt))

    events = self[jec_subjets](events, **kwargs)
    # if self.dataset_inst.is_mc:
    #     events = self[jer_subjets](events, **kwargs)
    events = self[msoftdrop](events, **kwargs)

    return events


@calibrator(
    uses={mc_weight, deterministic_seeds, jets_ak4, jets_ak8},
    produces={mc_weight, deterministic_seeds, jets_ak4, jets_ak8},
)
def no_jet_cleaning(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    events = self[jets_ak4](events, **kwargs)
    events = self[jets_ak8](events, **kwargs)  # call_force ?
    events = self[deterministic_seeds](events, **kwargs)

    return events
