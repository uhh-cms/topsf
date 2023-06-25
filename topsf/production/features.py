# coding: utf-8

"""
Column production methods related to higher-level features.
"""

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.production.util import attach_coffea_behavior

from topsf.production.lepton import choose_lepton
from topsf.production.gen_top import gen_top_decay
from topsf.production.probe_jet import probe_jet

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
        for junc_name in self.config_inst.x.jec.uncertainty_sources
        for junc_dir in ("up", "down")
    } | {"jer_up", "jer_down"}


@producer(
    uses={
        attach_coffea_behavior,
        choose_lepton,
        gen_top_decay,
        probe_jet,
    },
    produces={
        attach_coffea_behavior,
        choose_lepton,
        gen_top_decay,
        probe_jet,
    },
    shifts={
        jet_energy_shifts,
    },
)
def features(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """Producer for all high-level features."""

    # merge Muon and Electron columns to single Lepton column
    # depending on channel_id
    events = self[choose_lepton](self, **kwargs)

    # gen level top quark decay info
    events = self[gen_top_decay](self, **kwargs)

    # probe jet properties
    events = self[probe_jet](self, **kwargs)

    return events
