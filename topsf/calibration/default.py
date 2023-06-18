# coding: utf-8

"""
Calibration methods.
"""

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jets, jer, jec_nominal
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import


np = maybe_import("numpy")
ak = maybe_import("awkward")


@calibrator
def jets_skip_jecunc(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    """
    Common calibrator for Jet energy corrections, applying nominal JEC for data, and JEC with
    uncertainties plus JER for MC. Information about used and produced columns and dependent
    calibrators is added in a custom init function below.
    """
    if self.dataset_inst.is_mc:
        events = self[jec_nominal](events, **kwargs)
        events = self[jer](events, **kwargs)
    else:
        events = self[jec_nominal](events, **kwargs)

    return events


@jets_skip_jecunc.init
def jets_skip_jecunc_init(self: Calibrator) -> None:
    # add standard jec and jer for mc, and only jec nominal for dta
    if getattr(self, "dataset_inst", None) and self.dataset_inst.is_mc:
        self.uses |= {jec_nominal, jer}
        self.produces |= {jec_nominal, jer}
    else:
        self.uses |= {jec_nominal}
        self.produces |= {jec_nominal}


@calibrator(
    uses={mc_weight, deterministic_seeds, jets},
    produces={mc_weight, deterministic_seeds, jets},
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)
    events = self[deterministic_seeds](events, **kwargs)
    events = self[jets_skip_jecunc](events, **kwargs)

    return events
