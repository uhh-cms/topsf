# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior

ak = maybe_import("awkward")
coffea = maybe_import("coffea")
maybe_import("coffea.nanoevents.methods.nanoaod")


@producer
def jet_energy_shifts(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Pseudo-producer that registers jet energy shifts.
    """
    return events


@jet_energy_shifts.init
def jet_energy_shifts_init(self: Producer) -> None:
    """
    Register shifts.
    """
    self.shifts |= {
        f"jec_{junc_name}_{junc_dir}"
        for junc_name in self.config_inst.x.jec.Jet.uncertainty_sources
        for junc_dir in ("up", "down")
    } | {"jer_up", "jer_down"}


@producer(
    uses={
        attach_coffea_behavior,
        "event",
        "Jet.pt",
        "FatJet.pt",
        "Muon.pt",
        "Electron.pt",
    },
    produces={
        attach_coffea_behavior,
        "dummy",
        "n_jet",
        "n_fatjet",
        "n_muon",
        "n_electron",
    },
    shifts={
        jet_energy_shifts,
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """Producer for all high-level features."""

    # dummy to ensure at least one field
    events = set_ak_column(events, "dummy", ak.ones_like(events.event))

    # count jets and fatjets
    jet = ak.without_parameters(events["Jet"])
    fatjet = ak.without_parameters(events["FatJet"])
    muon = ak.without_parameters(events["Muon"])
    electron = ak.without_parameters(events["Electron"])
    events = set_ak_column(events, "n_jet", ak.num(jet.pt, axis=-1))
    events = set_ak_column(events, "n_fatjet", ak.num(fatjet.pt, axis=-1))
    events = set_ak_column(events, "n_muon", ak.num(muon.pt, axis=-1))
    events = set_ak_column(events, "n_electron", ak.num(electron.pt, axis=-1))

    return events
