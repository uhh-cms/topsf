# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.util import maybe_import

ak = maybe_import("awkward")


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    # (the "event", "run" and "lumi" variables are required for some cutflow plotting task,
    # and also correspond to the minimal set of columns that coffea's nano scheme requires)
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0, 1e9),
        x_title="Event ID",
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )

    # MC event weight
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )

    # Event properties
    config.add_variable(
        name="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
    )
    config.add_variable(
        name="n_bjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of b jets",
    )
    config.add_variable(
        name="n_lightjet",
        binning=(11, -0.5, 10.5),
        x_title="Number of light jets",
    )
    config.add_variable(
        name="n_electron",
        binning=(11, -0.5, 10.5),
        x_title="Number of electrons",
    )
    config.add_variable(
        name="n_muon",
        binning=(11, -0.5, 10.5),
        x_title="Number of muons",
    )

    # Object properties

    # Jets (4 pt-leading jets)
    for i in range(4):
        for obj in ("Jet", "FatJet"):
            config.add_variable(
                name=f"{obj.lower()}{i+1}_pt",
                expression=f"{obj}.pt[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, 0., 400.),
                unit="GeV",
                x_title=rf"{obj} {i+1} $p_{{T}}$",
            )
            config.add_variable(
                name=f"{obj.lower()}{i+1}_eta",
                expression=f"{obj}.eta[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(50, -2.5, 2.5),
                x_title=rf"{obj} {i+1} $\eta$",
            )
            config.add_variable(
                name=f"{obj.lower()}{i+1}_phi",
                expression=f"{obj}.phi[:,{i}]",
                null_value=EMPTY_FLOAT,
                binning=(40, -3.2, 3.2),
                x_title=rf"{obj} {i+1} $\phi$",
            )

    # Leptons
    for obj in ["Electron", "Muon"]:
        config.add_variable(
            name=f"{obj.lower()}_pt",
            expression=f"{obj}.pt[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, 0., 400.),
            unit="GeV",
            x_title=obj + r" $p_{T}$",
        )
        config.add_variable(
            name=f"{obj.lower()}_phi",
            expression=f"{obj}.phi[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=obj + r" $\phi$",
        )
        config.add_variable(
            name=f"{obj.lower()}_eta",
            expression=f"{obj}.eta[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(50, -2.5, 2.5),
            x_title=obj + r" $\eta$",
        )
        config.add_variable(
            name=f"{obj.lower()}_mass",
            expression=f"{obj}.mass[:,0]",
            null_value=EMPTY_FLOAT,
            binning=(40, -3.2, 3.2),
            x_title=obj + " mass",
        )

    # MET
    config.add_variable(
        name="met_pt",
        expression="MET.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0., 400.),
        unit="GeV",
        x_title=r"MET $p_{T}$",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, -3.2, 3.2),
        x_title=r"MET $\phi$",
    )

    # probe jet properties
    config.add_variable(
        name="probejet_pt",
        expression="ProbeJet.pt",
        null_value=EMPTY_FLOAT,
        #binning=(50, 0, 1000),  # noqa
        binning=(35, 300, 1000),
        x_title=r"Probe jet $p_{T}$",
        unit="GeV",
    )
    config.add_variable(
        name="probejet_mass",
        expression="ProbeJet.mass",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 500),
        x_title=r"Probe jet mass",
        unit="GeV",
    )
    config.add_variable(
        name="probejet_msoftdrop",
        expression="ProbeJet.msoftdrop",
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 500),
        x_title=r"Probe jet $m_{SD}$",
        unit="GeV",
    )
    config.add_variable(
        name="probejet_msoftdrop_widebins",
        expression="ProbeJet.msoftdrop",
        null_value=EMPTY_FLOAT,
        binning=[
            50, 70, 85,
            105, 120, 140,
            155, 170, 185,
            200, 210, 220,
            230, 250, 275,
            300, 350, 400,
            450, 500,
        ],
        x_title=r"Probe jet $m_{SD}$",
        unit="GeV",
    )
    config.add_variable(
        name="probejet_tau32",
        expression=lambda events: events.ProbeJet.tau3 / events.ProbeJet.tau2,
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 1),
        x_title=r"Probe jet $\tau_{3}/\tau_{2}$",
        aux={
            "inputs": {"ProbeJet.tau3", "ProbeJet.tau2"},
        },
    )
    config.add_variable(
        name="probejet_max_subjet_btag_score_btagDeepB",
        expression=lambda events: ak.max(
            events.ProbeJet.subjet_btag_scores_btagDeepB,
            axis=1,
        ),
        null_value=EMPTY_FLOAT,
        binning=(50, 0, 1),
        x_title=r"Max. DeepCSV score of probe subjets",
        log_y=True,
        aux={
            "inputs": {"ProbeJet.subjet_btag_scores_btagDeepB"},
        },
    )

    # jet lepton features
    config.add_variable(
        name="jet_lep_pt_rel",
        binning=(40, 0, 400),
        unit="GeV",
        x_title=r"$p_{T}^{rel}$(lepton, closest AK4 jet)",
    )
    config.add_variable(
        name="jet_lep_delta_r",
        binning=(40, 0, 5),
        x_title=r"$\Delta R$(lepton, closest AK4 jet)",
    )
    config.add_variable(
        name="jet_lep_pt_rel_zoom",
        expression="jet_lep_pt_rel",
        binning=(10, 0, 50),
        unit="GeV",
        x_title=r"$p_{T}^{rel}$(lepton, closest AK4 jet)",
    )
    config.add_variable(
        name="jet_lep_delta_r_zoom",
        expression="jet_lep_delta_r",
        binning=(15, 0, 1.5),
        x_title=r"$\Delta R$(lepton, closest AK4 jet)",
    )

    # HT: scalar jet pT sum
    config.add_variable(
        name="ht",
        expression=lambda events: ak.sum(events.Jet.pt, axis=1),
        binning=(40, 0.0, 800.0),
        unit="GeV",
        x_title="$H_{T}$",
        aux={
            "inputs": {"Jet.pt"},
        },
    )
