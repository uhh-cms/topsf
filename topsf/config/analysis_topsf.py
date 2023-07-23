# coding: utf-8

"""
Configuration of the topsf analysis.
"""

import functools
import law
import order as od
import os
import yaml

from order import Process
from scinum import Number

from columnflow.util import DotDict
from columnflow.config_util import (
    add_shift_aliases,
    get_root_processes_from_campaign,
    get_shifts_from_sources,
    verify_config_processes,
)

from topsf.config.variables import add_variables
from topsf.config.categories import add_categories

thisdir = os.path.dirname(os.path.abspath(__file__))

#
# the main analysis object
#

analysis_topsf = ana = od.Analysis(
    name="analysis_topsf",
    id=1,
)

# analysis-global versions
ana.x.versions = {}

# files of bash sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.bash_sandboxes = [
    "$CF_BASE/sandboxes/cf.sh",
    "$CF_BASE/sandboxes/venv_columnar.sh",
]

# files of cmssw sandboxes that might be required by remote tasks
# (used in cf.HTCondorWorkflow)
ana.x.cmssw_sandboxes = [
    # "$CF_BASE/sandboxes/cmssw_default.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("TOPSF_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}


#
# setup configs
#

# an example config is setup below, based on cms NanoAOD v9 for Run2 2017, focussing on
# ttbar and single top MCs, plus single muon data
# update this config or add additional ones to accomodate the needs of your analysis

from cmsdb.campaigns.run2_2017_nano_v9 import campaign_run2_2017_nano_v9

# copy the campaign
# (creates copies of all linked datasets, processes, etc. to allow for encapsulated customization)
campaign = campaign_run2_2017_nano_v9.copy()

# get all root processes
procs = get_root_processes_from_campaign(campaign)

# create parent processes for w_lep, dy_lep and qcd
for i_proc, (proc_name, proc_label, child_procs) in enumerate([
    ("vx", "V+jets, VV", ["dy_lep", "w_lnu", "vv"]),
    ("mj", "Multijet", ["qcd"]),
]):
    proc = Process(
        name=proc_name,
        id=int(1e7 + (i_proc + 1)),
        label=proc_label,
    )
    for child_proc in child_procs:
        procs.n(child_proc).add_parent_process(proc)

# get all root processes (including newly added ones)
procs = get_root_processes_from_campaign(campaign)

# create sub-processes for st, tt
# (defined via cuts on gen-level objects; will be normalized
# to xs of parent process)
top_subprocess_cfg = DotDict.wrap({
    "0o1q": {
        "index": 1,
        "label": "not merged (0q or 1q)",
        "colors": {
            "tt": "#A80068",
            "st": "#FF9300",
        },
    },
    "2q": {
        "index": 2,
        "label": "semi-merged (2q)",
        "colors": {
            "tt": "#FF58D0",
            "st": "#FFFF00",
        },
    },
    "3q": {
        "index": 3,
        "label": "fully merged (3q)",
        "colors": {
            "tt": "#FF0064",
            "st": "#FFC900",
        },
    },
    "bkg": {
        "index": 4,
        "label": "background",
        "colors": {
            "tt": "#700034",
            "st": "#A62800",
        },
    },
})


# helper function for adding subprocesses
def add_subprocesses(proc, color_key):
    """Add subprocesses to an existing process."""
    subprocs = {}
    for subproc_name, subproc_cfg in top_subprocess_cfg.items():
        subprocs[subproc_name] = subproc = proc.add_process(
            name=f"{proc.name}_{subproc_name}",
            id=int(proc.id + 1e6 * (subproc_cfg.index)),
            label=f"{proc.label}, {subproc_cfg.label}",
            color=subproc_cfg.colors[color_key],
            aux={
                "subprocess_name": subproc_name,
            },
        )
        subproc.add_tag("is_subprocess")
    proc.add_tag("has_subprocesses")
    return subprocs


# add subprocesses to processes with top quarks
for root_proc in ("st", "tt"):
    root_proc_inst = getattr(procs.n, root_proc)
    subprocs = {}  # [depth][subproc_key] -> od.Process
    for proc, depth, children in root_proc_inst.walk_processes(algo="bfs", include_self=True):
        # add subprocesses to top-level process (tt, st)
        subprocs[depth] = add_subprocesses(proc, color_key=root_proc)

        # mark subprocesses as children of parent subprocesses
        parent_subprocs = subprocs.get(depth - 1, {})
        if not parent_subprocs:
            continue
        for subproc_name, subproc_inst in subprocs[depth].items():
            parent_subprocs[subproc_name].add_process(subproc_inst)

# create a config by passing the campaign, so id and name will be identical
cfg = ana.add_config(campaign)

# gather campaign data
year = campaign.x.year

#
# processes
#

# set color of some processes
colors = {
    "data": "#000000",  # black
    "tt": "#E04F21",  # red
    "qcd": "#5E8FFC",  # blue
    "w_lnu": "#82FF28",  # green
    "st": "#3E00FB",  # dark purple
    "dy_lep": "#FBFF36",  # yellow
    "vv": "#B900FC",  # pink
    "other": "#999999",  # grey
    # christopher's color scheme
    "vx": "#00FF00",
    "mj": "#00D0FF",
}

# add processes we are interested in
process_names = [
    "data",
    "tt",
    "st",
    # "dy_lep",
    # "w_lnu",
    # "vv",
    # "qcd",
    "vx",
    "mj",
]
for process_name in process_names:
    # add the process
    proc = cfg.add_process(procs.get(process_name))

    # mark the presence of a top quark
    if any(proc.name.startswith(s) for s in ("tt", "st")):
        proc.add_tag("has_top")

    # mark ttbar processes (needed for top pt reweighting)
    if proc.name.startswith("tt"):
        proc.add_tag("is_ttbar")

    # configuration of colors, labels, etc. can happen here
    proc.color = colors.get(proc.name, "#aaaaaa")

#
# datasets
#

# add datasets we need to study
dataset_names = [
    # DATA
    "data_e_b",
    "data_e_c",
    "data_e_d",
    "data_e_e",
    "data_e_f",
    "data_mu_b",
    "data_mu_c",
    "data_mu_d",
    "data_mu_e",
    "data_mu_f",
    # TTbar
    "tt_sl_powheg",
    "tt_dl_powheg",
    "tt_fh_powheg",
    # SingleTop
    "st_schannel_lep_amcatnlo",
    "st_tchannel_t_powheg",
    "st_tchannel_tbar_powheg",
    "st_twchannel_t_powheg",
    "st_twchannel_tbar_powheg",
    # DY
    #"dy_lep_m50_ht70to100_madgraph",  # rm?  # noqa
    #"dy_lep_m50_ht100to200_madgraph",  # rm?  # noqa
    "dy_lep_m50_ht200to400_madgraph",
    "dy_lep_m50_ht400to600_madgraph",
    "dy_lep_m50_ht600to800_madgraph",
    "dy_lep_m50_ht800to1200_madgraph",
    "dy_lep_m50_ht1200to2500_madgraph",
    "dy_lep_m50_ht2500_madgraph",
    # WJets
    #"w_lnu_ht70To100_madgraph",  # rm?  # noqa
    #"w_lnu_ht100To200_madgraph",  # rm?  # noqa
    "w_lnu_ht200To400_madgraph",
    "w_lnu_ht400To600_madgraph",
    "w_lnu_ht600To800_madgraph",
    "w_lnu_ht800To1200_madgraph",
    "w_lnu_ht1200To2500_madgraph",
    "w_lnu_ht2500_madgraph",
    # Diboson
    "ww_pythia",
    "wz_pythia",
    "zz_pythia",
    # QCD
    #"qcd_ht50to100_madgraph",  # rm?  # noqa
    #"qcd_ht100to200_madgraph",  # rm?  # noqa
    #"qcd_ht200to300_madgraph",  # rm?  # noqa
    #"qcd_ht300to500_madgraph",  # rm?  # noqa
    "qcd_ht500to700_madgraph",
    "qcd_ht700to1000_madgraph",
    "qcd_ht1000to1500_madgraph",  # rm?
    "qcd_ht1500to2000_madgraph",
    "qcd_ht2000_madgraph",  # rm?
]
for dataset_name in dataset_names:
    # add the dataset
    dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

    # mark the presence of a top quark
    if any(dataset_name.startswith(s) for s in ("tt", "st")):
        dataset.add_tag("has_top")
        if "twchannel" in dataset_name:
            dataset.add_tag("has_top_associated_w")

    # mark ttbar (for top pT reweighting)
    if dataset_name.startswith("tt"):
        dataset.add_tag("is_ttbar")

    # mark v+jets processes (for NLO reweighting)
    if dataset_name.startswith("w_lnu"):
        dataset.add_tag("is_v_jets")
        dataset.add_tag("is_w_jets")
    if dataset_name.startswith("dy_lep"):
        dataset.add_tag("is_v_jets")
        dataset.add_tag("is_z_jets")

    # for testing purposes, limit the number of files to 1
    # for info in dataset.info.values():
    #     info.n_files = min(info.n_files, 1)

# verify that the root process of all datasets is part of any of the registered processes
verify_config_processes(cfg, warn=True)

# default objects, such as calibrator, selector, producer, ml model, inference model, etc
cfg.x.default_calibrator = "default"
cfg.x.default_selector = "default"
cfg.x.default_producer = "default"
cfg.x.default_ml_model = None
cfg.x.default_inference_model = "default"
cfg.x.default_categories = ("incl",)
cfg.x.default_variables = ("n_jet", "jet1_pt")

# process groups for conveniently looping over certain processs
# (used in wrapper_factory and during plotting)
cfg.x.process_groups = {}

# dataset groups for conveniently looping over certain datasets
# (used in wrapper_factory and during plotting)
cfg.x.dataset_groups = {
    "all": dataset_names,
    "data": ["data_*"],
    "dy_lep": ["dy_lep*"],
    "w_lnu": ["w_lnu*"],
    "qcd": ["qcd_ht*"],
    "st": ["st*"],
    "tt": ["tt*"],
    "vv": ["ww_pythia", "wz_pythia", "zz_pythia"],
}

# category groups for conveniently looping over certain categories
# (used during plotting)
cfg.x.category_groups = {}

# variable groups for conveniently looping over certain variables
# (used during plotting)
cfg.x.variable_groups = {}

# shift groups for conveniently looping over certain shifts
# (used during plotting)
cfg.x.shift_groups = {}

# selector step groups for conveniently looping over certain steps
# (used in cutflow tasks)
cfg.x.selector_step_groups = {
    "default": ["LeptonTrigger", "Lepton", "AddLeptonVeto", "MET", "BJet", "METFilters"],
}

# custom method and sandbox for determining dataset lfns
cfg.x.get_dataset_lfns = None
cfg.x.get_dataset_lfns_sandbox = None

# whether to validate the number of obtained LFNs in GetDatasetLFNs
# (currently set to false because the number of files per dataset is truncated to 2)
cfg.x.validate_dataset_lfns = False

# lumi values in inverse pb
# https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2?rev=2#Combination_and_correlations
cfg.x.luminosity = Number(41480, {
    "lumi_13TeV_2017": 0.02j,
    "lumi_13TeV_1718": 0.006j,
    "lumi_13TeV_correlated": 0.009j,
})

# MET filters
# https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2?rev=158#2018_2017_data_and_MC_UL
cfg.x.met_filters = {
    "Flag.goodVertices",
    "Flag.globalSuperTightHalo2016Filter",
    "Flag.HBHENoiseFilter",
    "Flag.HBHENoiseIsoFilter",
    "Flag.EcalDeadCellTriggerPrimitiveFilter",
    "Flag.BadPFMuonFilter",
    "Flag.BadPFMuonDzFilter",
    "Flag.eeBadScFilter",
    "Flag.ecalBadCalibFilter",
}

# names of muon correction sets and working points
# (used in the muon producer)
cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", f"{year}_UL")

# location of JEC txt files
cfg.x.jec = DotDict.wrap({
    "campaign": "Summer19UL17",
    "version": "V5",
    "jet_type": "AK4PFchs",
    "levels": ["L1L2L3Res"],
    "levels_for_type1_met": ["L1FastJet"],
    "data_eras": sorted(filter(None, {
        d.x("jec_era", None)
        for d in cfg.datasets
        if d.is_data
    })),
    "uncertainty_sources": [
        # comment out most for now to prevent large file sizes
        # "AbsoluteStat",
        # "AbsoluteScale",
        # "AbsoluteSample",
        # "AbsoluteFlavMap",
        # "AbsoluteMPFBias",
        # "Fragmentation",
        # "SinglePionECAL",
        # "SinglePionHCAL",
        # "FlavorQCD",
        # "TimePtEta",
        # "RelativeJEREC1",
        # "RelativeJEREC2",
        # "RelativeJERHF",
        # "RelativePtBB",
        # "RelativePtEC1",
        # "RelativePtEC2",
        # "RelativePtHF",
        # "RelativeBal",
        # "RelativeSample",
        # "RelativeFSR",
        # "RelativeStatFSR",
        # "RelativeStatEC",
        # "RelativeStatHF",
        # "PileUpDataMC",
        # "PileUpPtRef",
        # "PileUpPtBB",
        # "PileUpPtEC1",
        # "PileUpPtEC2",
        # "PileUpPtHF",
        # "PileUpMuZero",
        # "PileUpEnvelope",
        # "SubTotalPileUp",
        # "SubTotalRelative",
        # "SubTotalPt",
        # "SubTotalScale",
        # "SubTotalAbsolute",
        # "SubTotalMC",
        "Total",
        # "TotalNoFlavor",
        # "TotalNoTime",
        # "TotalNoFlavorNoTime",
        # "FlavorZJet",
        # "FlavorPhotonJet",
        # "FlavorPureGluon",
        # "FlavorPureQuark",
        # "FlavorPureCharm",
        # "FlavorPureBottom",
        # "TimeRunA",
        # "TimeRunB",
        # "TimeRunC",
        # "TimeRunD",
        #
        # "CorrelationGroupMPFInSitu",
        # "CorrelationGroupIntercalibration",
        # "CorrelationGroupbJES",
        # "CorrelationGroupFlavor",
        # "CorrelationGroupUncorrelated",
    ],
})

cfg.x.jer = DotDict.wrap({
    "campaign": "Summer19UL17",
    "version": "JRV2",
    "jet_type": "AK4PFchs",
})


# names of electron correction sets and working points
# (used in the electron_sf producer)
# TODO: check that these are appropriate
cfg.x.electron_sf_names = ("UL-Electron-ID-SF", "2017", "wp80iso")

# names of muon correction sets and working points
# (used in the muon producer)
# TODO: check that these are appropriate
cfg.x.muon_sf_names = ("NUM_TightRelIso_DEN_TightIDandIPCut", "2017_UL")

# JEC uncertainty sources propagated to btag scale factors
# (names derived from contents in BTV correctionlib file)
year = cfg.campaign.x.year
cfg.x.btag_sf_jec_sources = [
    "",  # same as "Total"
    # "Absolute",
    # "AbsoluteMPFBias",
    # "AbsoluteScale",
    # "AbsoluteStat",
    # f"Absolute_{year}",
    # "BBEC1",
    # f"BBEC1_{year}",
    # "EC2",
    # f"EC2_{year}",
    # "FlavorQCD",
    # "Fragmentation",
    # "HF",
    # f"HF_{year}",
    # "PileUpDataMC",
    # "PileUpPtBB",
    # "PileUpPtEC1",
    # "PileUpPtEC2",
    # "PileUpPtHF",
    # "PileUpPtRef",
    # "RelativeBal",
    # "RelativeFSR",
    # "RelativeJEREC1",
    # "RelativeJEREC2",
    # "RelativeJERHF",
    # "RelativePtBB",
    # "RelativePtEC1",
    # "RelativePtEC2",
    # "RelativePtHF",
    # "RelativeSample",
    # f"RelativeSample_{year}",
    # "RelativeStatEC",
    # "RelativeStatFSR",
    # "RelativeStatHF",
    # "SinglePionECAL",
    # "SinglePionHCAL",
    # "TimePtEta",
]

# name of the btag_sf correction set and jec uncertainties to propagate through
cfg.x.btag_sf = ("deepJet_shape", cfg.x.btag_sf_jec_sources)

# L1 prefiring configuration
cfg.x.l1_prefiring = DotDict.wrap({
    "jet": {
        "value": "l1_prefiring_efficiency_value_jetpt_2017BtoF",
        "error": "l1_prefiring_efficiency_error_jetpt_2017BtoF",
    },
    "photon": {
        "value": "l1_prefiring_efficiency_value_photonpt_2017BtoF",
        "error": "l1_prefiring_efficiency_error_photonpt_2017BtoF",
    },
})

# V+jets reweighting
cfg.x.vjets_reweighting = DotDict.wrap({
    "w": {
        "value": "wjets_kfactor_value",
        "error": "wjets_kfactor_error",
    },
    "z": {
        "value": "zjets_kfactor_value",
        "error": "zjets_kfactor_error",
    },
})

#
# register shifts
#

# read in JEC sources from file
with open(os.path.join(thisdir, "jec_sources.yaml"), "r") as f:
    all_jec_sources = yaml.load(f, yaml.Loader)["names"]


# declare the shifts
def add_shifts(cfg):
    # nominal shift
    cfg.add_shift(name="nominal", id=0)

    # tune shifts are covered by dedicated, varied datasets, so tag the shift as "disjoint_from_nominal"
    # (this is currently used to decide whether ML evaluations are done on the full shifted dataset)
    cfg.add_shift(name="tune_up", id=1, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="tune_down", id=2, type="shape", tags={"disjoint_from_nominal"})

    cfg.add_shift(name="hdamp_up", id=3, type="shape", tags={"disjoint_from_nominal"})
    cfg.add_shift(name="hdamp_down", id=4, type="shape", tags={"disjoint_from_nominal"})

    # pileup / minimum bias cross section variations
    cfg.add_shift(name="minbias_xs_up", id=7, type="shape")
    cfg.add_shift(name="minbias_xs_down", id=8, type="shape")
    add_shift_aliases(cfg, "minbias_xs", {"pu_weight": "pu_weight_{name}"})

    # top pt reweighting
    cfg.add_shift(name="top_pt_up", id=9, type="shape")
    cfg.add_shift(name="top_pt_down", id=10, type="shape")
    add_shift_aliases(cfg, "top_pt", {"top_pt_weight": "top_pt_weight_{direction}"})

    # renormalization scale
    cfg.add_shift(name="mur_up", id=101, type="shape")
    cfg.add_shift(name="mur_down", id=102, type="shape")

    # factorization scale
    cfg.add_shift(name="muf_up", id=103, type="shape")
    cfg.add_shift(name="muf_down", id=104, type="shape")

    # scale variation (?)
    cfg.add_shift(name="scale_up", id=105, type="shape")
    cfg.add_shift(name="scale_down", id=106, type="shape")

    # pdf variations
    cfg.add_shift(name="pdf_up", id=107, type="shape")
    cfg.add_shift(name="pdf_down", id=108, type="shape")

    # alpha_s variation
    cfg.add_shift(name="alpha_up", id=109, type="shape")
    cfg.add_shift(name="alpha_down", id=110, type="shape")

    for unc in ["mur", "muf", "scale", "pdf", "alpha"]:
        add_shift_aliases(cfg, unc, {f"{unc}_weight": unc + "_weight_{direction}"})

    # event weights due to muon scale factors
    cfg.add_shift(name="muon_up", id=111, type="shape")
    cfg.add_shift(name="muon_down", id=112, type="shape")
    add_shift_aliases(cfg, "muon", {"muon_weight": "muon_weight_{direction}"})

    # event weights due to electron scale factors
    cfg.add_shift(name="electron_up", id=113, type="shape")
    cfg.add_shift(name="electron_down", id=114, type="shape")
    add_shift_aliases(cfg, "electron", {"electron_weight": "electron_weight_{direction}"})

    # V+jets reweighting
    cfg.add_shift(name="vjets_up", id=201, type="shape")
    cfg.add_shift(name="vjets_down", id=202, type="shape")
    add_shift_aliases(cfg, "vjets", {"vjets_weight": "vjets_weight_{direction}"})

    # prefiring weights
    cfg.add_shift(name="l1_prefiring_up", id=301, type="shape")
    cfg.add_shift(name="l1_prefiring_down", id=302, type="shape")
    add_shift_aliases(cfg, "l1_prefiring", {"l1_prefiring_weight": "l1_prefiring_weight_{direction}"})

    # jet energy scale (JEC) uncertainty variations
    for jec_source in cfg.x.jec.uncertainty_sources:
        idx = all_jec_sources.index(jec_source)
        cfg.add_shift(name=f"jec_{jec_source}_up", id=5000 + 2 * idx, type="shape")
        cfg.add_shift(name=f"jec_{jec_source}_down", id=5001 + 2 * idx, type="shape")
        add_shift_aliases(
            cfg,
            f"jec_{jec_source}",
            {
                "Jet.pt": "Jet.pt_{name}",
                "Jet.mass": "Jet.mass_{name}",
                "MET.pt": "MET.pt_{name}",
            },
        )

    # jet energy resolution (JER) scale factor variations
    cfg.add_shift(name="jer_up", id=6000, type="shape")
    cfg.add_shift(name="jer_down", id=6001, type="shape")
    add_shift_aliases(
        cfg,
        "jer",
        {
            "Jet.pt": "Jet.pt_{name}",
            "Jet.mass": "Jet.mass_{name}",
            "MET.pt": "MET.pt_{name}",
        },
    )


# add the shifts
add_shifts(cfg)

# external files
json_mirror = "/afs/cern.ch/user/m/mrieger/public/mirrors/jsonpog-integration-dfd90038"
cfg.x.external_files = DotDict.wrap({
    # jet energy corrections
    "jet_jerc": (f"{json_mirror}/POG/JME/{year}_UL/jet_jerc.json.gz", "v1"),  # noqa

    # btag scale factors
    "btag_sf_corr": (f"{json_mirror}/POG/BTV/{year}_UL/btagging.json.gz", "v1"),  # noqa

    # electron scale factors
    "electron_sf": (f"{json_mirror}/POG/EGM/{year}_UL/electron.json.gz", "v1"),  # noqa

    # muon scale factors
    "muon_sf": (f"{json_mirror}/POG/MUO/{year}_UL/muon_Z.json.gz", "v1"),

    # L1 prefiring corrections
    "l1_prefiring": f"{os.getenv('TOPSF_BASE')}/data/json/l1_prefiring.json.gz",

    # V+jets reweighting
    "vjets_reweighting": f"{os.getenv('TOPSF_BASE')}/data/json/vjets_reweighting.json.gz",

    # lumi files
    "lumi": {
        "golden": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt", "v1"),  # noqa
        "normtag": ("/afs/cern.ch/user/l/lumipro/public/Normtags/normtag_PHYSICS.json", "v1"),
    },

    # files from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData?rev=44#Pileup_JSON_Files_For_Run_II
    "pu": {
        "json": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/pileup_latest.txt", "v1"),  # noqa
        "mc_profile": ("https://raw.githubusercontent.com/cms-sw/cmssw/435f0b04c0e318c1036a6b95eb169181bbbe8344/SimGeneral/MixingModule/python/mix_2017_25ns_UltraLegacy_PoissonOOTPU_cfi.py", "v1"),  # noqa
        "data_profile": {
            "nominal": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root", "v1"),  # noqa
            "minbias_xs_up": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root", "v1"),  # noqa
            "minbias_xs_down": ("/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root", "v1"),  # noqa
        },
    },
})

# target file size after MergeReducedEvents in MB
cfg.x.reduced_file_size = 512.0

# columns to keep after certain steps
cfg.x.keep_columns = DotDict.wrap({
    "cf.ReduceEvents": {
        #
        # NanoAOD columns
        #

        # general event info
        "run", "luminosityBlock", "event",

        # weights
        "genWeight",
        "LHEWeight.*",
        "LHEPdfWeight", "LHEScaleWeight",

        # muons
        "Muon.pt", "Muon.eta", "Muon.phi", "Muon.mass",
        "Muon.pdgId",
        "Muon.jetIdx",
        "Muon.nStations",
        "Muon.pfRelIso03_all", "Muon.pfRelIso04_all",

        # electrons
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.pdgId",
        "Electron.jetIdx",
        "Electron.deltaEtaSC",
        "Electron.pfRelIso03_all",

        # photons (for L1 prefiring)
        "Photon.pt", "Photon.eta", "Photon.phi", "Photon.mass",
        "Photon.jetIdx",

        # AK4 jets
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Jet.rawFactor",
        "Jet.btagDeepFlavB", "Jet.hadronFlavour",
        # # optional, enable if needed
        # "Jet.area",
        # "Jet.hadronFlavour", "Jet.partonFlavour",
        # "Jet.jetId", "Jet.puId", "Jet.puIdDisc",
        # # cleaning
        # "Jet.cleanmask",
        # "Jet.muonSubtrFactor",
        # # indices to other collections
        # "Jet.electronIdx*",
        # "Jet.muonIdx*",
        # "Jet.genJetIdx*",
        # # number of jet constituents
        # "Jet.nConstituents",
        # "Jet.nElectrons",
        # "Jet.nMuons",
        # # PF energy fractions
        # "Jet.chEmEF",
        # "Jet.chHEF",
        # "Jet.neEmEF",
        # "Jet.neHEF",
        # "Jet.muEF",
        # # taggers
        # "Jet.qgl",
        # "Jet.btag*",

        # AK8 jets
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass", "FatJet.msoftdrop",
        "FatJet.rawFactor",
        "FatJet.tau1", "FatJet.tau2", "FatJet.tau3", "FatJet.tau4",
        "FatJet.subJetIdx1", "FatJet.subJetIdx2",
        # # optional, enable if needed
        # "FatJet.area", "FatJet.jetId", "FatJet.hadronFlavour",
        # "FatJet.genJetAK8Idx",
        # "FatJet.muonIdx3SJ", "FatJet.electronIdx3SJ",
        # "FatJet.nBHadrons", "FatJet.nCHadrons",
        # # taggers
        # "FatJet.btag*", "FatJet.deepTag*", "FatJet.particleNet*",

        # subjets
        "SubJet.btagDeepB",

        # generator quantities
        "Generator.*",

        # generator particles
        # "GenPart.pt", "GenPart.eta", "GenPart.phi", "GenPart.mass",
        # "GenPart.pdgId",
        # "GenPart.*",

        # missing transverse momentum
        "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",

        # number of primary vertices
        "PV.npvs",

        #
        # columns added during selection
        #

        # generator particle info
        "GenTopDecay.*",
        "GenTopAssociatedDecay.*",
        "GenPartonTop.*",
        "GenVBoson.*",

        # generic leptons (merger of Muon/Electron)
        "Lepton.*",

        # probe jet
        "ProbeJet.*",

        # columns for PlotCutflowVariables
        "cutflow.*",

        # others
        "channel_id", "category_ids", "process_id",
        "deterministic_seed",
        "mc_weight",
        "pu_weight*",
    },
    "cf.MergeSelectionMasks": {
        "channel_id", "process_id", "category_ids",
        "normalization_weight",
        "cutflow.*",
    },
    "cf.UniteColumns": {
        "*",
    },
})

# event weight columns as keys in an OrderedDict, mapped to shift instances they depend on
get_shifts = functools.partial(get_shifts_from_sources, cfg)
cfg.x.event_weights = DotDict({
    "normalization_weight": [],
    "muon_weight": get_shifts("muon"),
    "electron_weight": get_shifts("electron"),
    "top_pt_weight": get_shifts("top_pt"),
    "l1_prefiring_weight": get_shifts("l1_prefiring"),
    "vjets_weight": get_shifts("vjets"),
})

# named references to actual versions to use for certain sets of tasks
main_ver = "v7"
cfg.x.named_versions = DotDict.wrap({
    "default": f"{main_ver}",
    "calibrate": "v2",
    "select": "v6",
    "reduce": f"{main_ver}",
    "merge": f"{main_ver}",
    "produce": f"{main_ver}",
    "hist": f"{main_ver}",
    "plot": f"{main_ver}",
    "datacards": f"{main_ver}",
})

# versions per task family and optionally also dataset and shift
# None can be used as a key to define a default value
cfg.x.versions = {
    None: cfg.x.named_versions["default"],
    # CSR tasks
    "cf.CalibrateEvents": cfg.x.named_versions["calibrate"],
    "cf.SelectEvents": cfg.x.named_versions["select"],
    "cf.ReduceEvents": cfg.x.named_versions["reduce"],
    # merging tasks
    "cf.MergeSelectionStats": cfg.x.named_versions["merge"],
    "cf.MergeSelectionMasks": cfg.x.named_versions["merge"],
    "cf.MergeReducedEvents": cfg.x.named_versions["merge"],
    "cf.MergeReductionStats": cfg.x.named_versions["merge"],
    # column production
    "cf.ProduceColumns": cfg.x.named_versions["produce"],
    # histogramming
    "cf.CreateCutflowHistograms": cfg.x.named_versions["hist"],
    "cf.CreateHistograms": cfg.x.named_versions["hist"],
    "cf.MergeHistograms": cfg.x.named_versions["hist"],
    "cf.MergeShiftedHistograms": cfg.x.named_versions["hist"],
    # plotting
    "cf.PlotVariables1D": cfg.x.named_versions["plot"],
    "cf.PlotVariables2D": cfg.x.named_versions["plot"],
    "cf.PlotVariablesPerProcess2D": cfg.x.named_versions["plot"],
    "cf.PlotShiftedVariables1D": cfg.x.named_versions["plot"],
    "cf.PlotShiftedVariablesPerProcess1D": cfg.x.named_versions["plot"],
    #
    "cf.PlotCutflow": cfg.x.named_versions["plot"],
    "cf.PlotCutflowVariables1D": cfg.x.named_versions["plot"],
    "cf.PlotCutflowVariables2D": cfg.x.named_versions["plot"],
    "cf.PlotCutflowVariablesPerProcess2D": cfg.x.named_versions["plot"],
    # datacards
    "cf.CreateDatacards": cfg.x.named_versions["datacards"],
}

# top pt reweighting parameters
# https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#TOP_PAG_corrections_based_on_dat?rev=31
cfg.x.top_pt_reweighting_params = {
    "a": 0.0615,
    "b": -0.0005,
}

# 2017 b-tag working points
# https://twiki.cern.ch/twiki/bin/view/CMS/BtagRecommendation106XUL17?rev=15
cfg.x.btag_working_points = DotDict.wrap({
    "deepjet": {
        "loose": 0.0532,
        "medium": 0.3040,
        "tight": 0.7476,
    },
    "deepcsv": {
        "loose": 0.1355,
        "medium": 0.4506,
        "tight": 0.7738,
    },
})

# top-tag working points
# https://twiki.cern.ch/twiki/bin/view/CMS/JetTopTagging?rev=41
cfg.x.toptag_working_points = DotDict.wrap({
    "tau32": {
        "very_loose": 0.69,
        "loose": 0.61,
        "medium": 0.52,
        "tight": 0.47,
        "very_tight": 0.38,
    },
})

# lepton selection parameters
cfg.x.lepton_selection = DotDict.wrap({
    "mu": {
        "column": "Muon",
        "min_pt": 55,
        "max_abseta": 2.4,
        "triggers": {
            # # FIXME: adapt for UL17
            # "IsoMu24",
            # "IsoTkMu24",
            # as mttbar:
            "IsoMu27",
        },
        "id": {
            "column": "tightId",
            "value": True,
        },
        "rel_iso": "pfRelIso03_all",
        "max_rel_iso": 1.5,
        # veto events with additional leptons passing looser cuts
        "min_pt_addveto": 30,
        "id_addveto": {
            "column": "looseId",
            "value": True,
        },
    },
    "e": {
        "column": "Electron",
        "min_pt": 55,
        "max_abseta": 2.4,
        "triggers": {
            # FIXME: adapt for UL17
            # "Ele27_WPTight_Gsf",
            # "Ele115_CaloIdVT_GsfTrkIdT",
            # as mttbar:
            "Ele35_WPTight_Gsf",
        },
        #"id": "mvaFall17V2Iso_WP90",  # noqa
        "id": {
            "column": "cutBased",
            "value": 4,
        },
        # veto events with additional leptons passing looser cuts
        "min_pt_addveto": 30,
        #"id_addveto": "mvaFall17V2Iso_WPL",  # noqa
        "id_addveto": {
            "column": "cutBased",
            "value": 1,
        },
    },
})

# jet selection parameters
cfg.x.jet_selection = DotDict.wrap({
    "ak8": {
        "column": "FatJet",
        "min_pt": 300,
        "max_abseta": 2.5,
        "msoftdrop_range": (105, 210),
        # probe jet pt bins )used by category builder)
        "pt_bins": [300, 400, 480, 600, None],
        # parameters for b-tagged subjets
        "subjet_column": "SubJet",
        "subjet_btag": "btagDeepB",
        "subjet_btag_wp": cfg.x.btag_working_points.deepcsv.loose,
    },
    # TODO: implement (requires custom nano)
    "hotvr": {
        "column": "HOTVRJetForTopTagging",
        "min_pt": 200,
        "max_abseta": 2.5,
        # clustering parameters (not needed for analysis, added for reference)
        # https://twiki.cern.ch/twiki/bin/view/CMS/JetTopTagging?rev=41
        "r_min_max": (0.1, 1.5),
        "rho": 600,  # GeV
        "mu": 30,  # Gev, mass jump threshold
        "theta": 0.7,  # mass jump strength
        "min_pt_subjet": 30,  # min pt of subject
    },
    "ak4": {
        "column": "Jet",
        "min_pt": 15,  # TODO: check UHH2
        "max_abseta": 2.5,  # TODO: check UHH2
        "btag_column": "btagDeepFlavB",  # nano v9: "DeepJet b+bb+lepb tag discriminator"
        "btag_wp": cfg.x.btag_working_points.deepjet.medium,
    },
})

# MET selection parameters
cfg.x.met_selection = DotDict.wrap({
    "default": {
        "column": "MET",
        "min_pt": 50,
    },
})

# add categories
add_categories(cfg)

# add variables
add_variables(cfg)

# add channels
cfg.add_channel("e", id=1)
cfg.add_channel("mu", id=2)
