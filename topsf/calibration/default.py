# coding: utf-8

"""
Calibration methods.
"""
import functools
import law

from columnflow.calibration import Calibrator, calibrator
from columnflow.calibration.cms.jets import jec_ak4, jer_ak4, jec_ak8, jer_ak8
from columnflow.calibration.cms.met import met_phi
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.production.cms.jet import msoftdrop
from columnflow.production.cms.seeds import deterministic_seeds
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.cms.jet import jet_id, fatjet_id

from topsf.calibration.jets import (
    jet_lepton_cleaner,
    jec_subjets,
    jer_subjets
)
from topsf.util import has_tag, record_calls

logger = law.logger.get_logger(__name__)

np = maybe_import("numpy")
ak = maybe_import("awkward")

set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

jec_ak4_Puppi = jec_ak4.derive(
    "jec_ak4_Puppi",
    cls_dict={
        "met_name": "PuppiMET",
        "raw_met_name": "RawPuppiMET",
    }
)
jer_ak4_Puppi = jer_ak4.derive(
    "jer_ak4_Puppi",
    cls_dict={
        "met_name": "PuppiMET",
        "raw_met_name": "RawPuppiMET",
    }
)

jec_ak8_Puppi = jec_ak8.derive(
    "jec_ak8_Puppi",
    cls_dict={
        "propagate_met": False,
        "met_name": "DO_NOT_USE",
        "raw_met_name": "DO_NOT_USE",
    },
)

jer_ak8_Puppi = jer_ak8.derive(
    "jer_ak8_Puppi",
    cls_dict={
        "propagate_met": False,
        "met_name": "DO_NOT_USE",
        "raw_met_name": "DO_NOT_USE",
    }
)

jec_subjets_Puppi = jec_subjets.derive(
    "jec_subjets_Puppi",
    cls_dict={
        "propagate_met": False,
        "met_name": "DO_NOT_USE",
        "raw_met_name": "DO_NOT_USE",
    }
)

jer_subjets_Puppi = jer_subjets.derive(
    "jer_subjets_Puppi",
    cls_dict={
        "propagate_met": False,
        "met_name": "DO_NOT_USE",
        "raw_met_name": "DO_NOT_USE",
    }
)


@calibrator(
    uses={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        msoftdrop,
        "Muon.pt", "Muon.tunepRelPt",
    },
    produces={
        mc_weight,
        deterministic_seeds,
        jet_lepton_cleaner,
        msoftdrop,
        "Muon.pt", "Muon.rawPt",
    },
)
def default(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    run_list = []
    with record_calls(self, run_list):
        # highPt muons: use tuneP pT
        events = set_ak_column_f32(events, "Muon.rawPt", events.Muon.pt)
        events = set_ak_column_f32(events, "Muon.pt", events.Muon.tunepRelPt * events.Muon.pt)
        logger.info_once(
            "Finished recalculating muon pt with tuneP for highPt muons. Stored original pt in Muon.rawPt."
        )
        if self.dataset_inst.is_mc:
            events = self[mc_weight](events, **kwargs)
        events = self[deterministic_seeds](events, **kwargs)

        events = self[jet_lepton_cleaner](events, **kwargs)  # set Jet.pt to raw Pt -> run before jets (JER) calibrator
        # run JEC calibrators for AK4 and AK8 jets
        # fake subjet area column by setting it to an array with the same structure as the subjet pt column containing 0.5
        # (needed to be able to use same code as for top-level AK4/AK8 jets, as the producer formally requires an `area`
        # column, despite not actually using it)
        events = set_ak_column_f32(events, "SubJet.area", 0.5 * ak.ones_like(events.SubJet.pt))
        if self.config_inst.x.year == 2024:
            events = self[jec_ak4_Puppi](events, **kwargs)
            events = self[jec_ak8_Puppi](events, **kwargs)
            events = self[jec_subjets_Puppi](events, **kwargs)
            if self.dataset_inst.is_mc:
                events = self[jer_ak4_Puppi](events, **kwargs)
                events = self[jer_ak8_Puppi](events, **kwargs)
                # events = self[jer_subjets_Puppi](events, **kwargs)
        else:
            events = self[jec_ak4](events, **kwargs)
            events = self[jec_ak8](events, **kwargs)
            events = self[jec_subjets](events, **kwargs)
            if self.dataset_inst.is_mc:
                events = self[jer_ak4](events, **kwargs)
                events = self[jer_ak8](events, **kwargs)
                # events = self[jer_subjets](events, **kwargs)

        events = self[msoftdrop](events, **kwargs)
        if self.config_inst.x.year in {2022, 2023}:
            events = self[met_phi](events, **kwargs)
        elif self.config_inst.x.year == 2024:
            logger.warning_once("met_phi calibrator not run for 2024 config, as it is not yet available.")
        else:
            raise ValueError(f"Unsupported year {self.config_inst.x.year} in default calibrator")

        if not has_tag("skip_jet_ids", self.config_inst, self.dataset_inst, operator=any):
            logger.debug("Recalulating (fat)jet IDs.")
            events = self[jet_id](events, **kwargs)
            events = self[fatjet_id](events, **kwargs)

    logger.info_once(
        "Finished default calibration steps:\n" +
        "\n".join(run_list)
    )

    return events


@default.init
def default_init(self: Calibrator) -> None:
    # add met_phi only for 2022/23 configs
    if self.config_inst.x.year in {2022, 2023}:
        self.uses |= {
            met_phi,
            jec_ak4,
            jec_ak8,
            jer_ak4,
            jer_ak8,
            jec_subjets,
            jer_subjets,
        }
        self.produces |= {
            met_phi,
            jec_ak4,
            jec_ak8,
            jer_ak4,
            jer_ak8,
            jec_subjets,
            jer_subjets,
        }
    elif self.config_inst.x.year == 2024:
        self.uses |= {
            jec_ak4_Puppi,
            jec_ak8_Puppi,
            jer_ak4_Puppi,
            jer_ak8_Puppi,
            jec_subjets_Puppi,
            jer_subjets_Puppi,
        }
        self.produces |= {
            jec_ak4_Puppi,
            jec_ak8_Puppi,
            jer_ak4_Puppi,
            jer_ak8_Puppi,
            jec_subjets_Puppi,
            jer_subjets_Puppi,
        }

    if not has_tag("skip_jet_ids", self.config_inst, self.dataset_inst, operator=any):
        self.uses |= {
            jet_id,
            fatjet_id,
        }
        self.produces |= {
            jet_id,
            fatjet_id,
        }


@calibrator(
    uses={mc_weight, deterministic_seeds},
    produces={mc_weight, deterministic_seeds},
)
def no_jet_cleaning(self: Calibrator, events: ak.Array, **kwargs) -> ak.Array:
    if self.dataset_inst.is_mc:
        events = self[mc_weight](events, **kwargs)

    # events = self[jets_ak4](events, **kwargs)
    # events = self[jets_ak8](events, **kwargs)  # call_force ?
    events = self[deterministic_seeds](events, **kwargs)

    return events
