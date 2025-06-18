
# coding: utf-8

"""
Producers related to event weights.
"""

from columnflow.production import Producer, producer
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.muon import muon_weights
from columnflow.production.cms.pileup import pu_weight
from columnflow.util import maybe_import

from topsf.production.normalization import normalization_weights
from topsf.production.gen_top import top_pt_weight
from topsf.production.gen_v import vjets_weight
from topsf.production.ps_weights import ps_weights
from topsf.production.l1_prefiring import l1_prefiring_weights
from topsf.util import has_tag

ak = maybe_import("awkward")


@producer
def weights(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Main event weight producer (e.g. MC generator, scale factors, normalization).
    """
    if self.dataset_inst.is_mc:
        if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
            electron_mask = (events.Electron.pt >= 35)
            events = self[electron_weights](events, electron_mask=electron_mask, **kwargs)
        if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
            muon_mask = (events.Muon.pt >= 30) & (abs(events.Muon.eta) < 2.4)
            events = self[muon_weights](events, muon_mask=muon_mask, **kwargs)

        # # compute btag weights
        # jet_mask = (events.Jet.pt >= 100) & (abs(events.Jet.eta) < 2.5)
        # events = self[btag_weights](events, jet_mask=jet_mask, **kwargs)

        # compute top pT weights
        if self.dataset_inst.has_tag("is_ttbar"):
            events = self[top_pt_weight](events, **kwargs)

        # compute V+jets K factor weights
        if self.dataset_inst.has_tag("is_v_jets"):
            events = self[vjets_weight](events, **kwargs)

        if self.config_inst.x.run == 2:
            # compute L1 prefiring weights
            events = self[l1_prefiring_weights](events, **kwargs)

        # compute normalization weights
        events = self[normalization_weights](events, **kwargs)

        # compute MC weights
        events = self[mc_weight](events, **kwargs)

        # compute pu weights
        events = self[pu_weight](events, **kwargs)

        # compute PS weights
        events = self[ps_weights](events, **kwargs)

    return events


@weights.init
def weights_init(self: Producer) -> None:
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_mc:
        # dynamically add dependencies if running on MC
        if not has_tag("skip_electron_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {electron_weights}
            self.produces |= {electron_weights}

        if not has_tag("skip_muon_weights", self.config_inst, self.dataset_inst, operator=any):
            self.uses |= {muon_weights}
            self.produces |= {muon_weights}

        if self.config_inst.x.run == 2:
            self.uses |= {l1_prefiring_weights}
            self.produces |= {l1_prefiring_weights}

        self.uses |= {
            normalization_weights, pu_weight, mc_weight,
            top_pt_weight,
            vjets_weight,
            ps_weights,
        }
        self.produces |= {
            normalization_weights, pu_weight, mc_weight,
            top_pt_weight,
            vjets_weight,
            ps_weights,
        }
