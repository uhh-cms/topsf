# coding: utf-8

"""
Configuration of the topsf analysis.
"""

import functools
import law
import order as od
import os
import yaml

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
    "$CF_BASE/sandboxes/cf_prod.sh",
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

# create a config by passing the campaign, so id and name will be identical
cfg = ana.add_config(campaign)

# gather campaign data
year = campaign.x.year

# set color of some processes
colors = {
    "data": "#000000",  # black
    "tt": "#E04F21",  # red
    "qcd": "#5E8FFC",  # blue
    "w_lnu": "#82FF28",  # green
    "higgs": "#984ea3",  # purple
    "st": "#3E00FB",  # dark purple
    "dy_lep": "#FBFF36",  # yellow
    "vv": "#B900FC",  # pink
    "other": "#999999",  # grey
}

# add processes we are interested in
process_names = [
    "data",
    "tt",
    "st",
    "dy_lep",
    "w_lnu",
    "vv",
    "qcd",
]
for process_name in process_names:
    # add the process
    proc = cfg.add_process(procs.get(process_name))

    # mark the presence of a top quark
    if any(proc.name.startswith(s) for s in ("tt", "st")):
        proc.add_tag("has_top")

    # configuration of colors, labels, etc. can happen here
    proc.color1 = colors.get(proc.name, "#aaaaaa")
    proc.color2 = colors.get(proc.name, "#000000")

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
    "dy_lep_m50_ht70to100_madgraph",
    "dy_lep_m50_ht100to200_madgraph",
    "dy_lep_m50_ht200to400_madgraph",
    "dy_lep_m50_ht400to600_madgraph",
    "dy_lep_m50_ht600to800_madgraph",
    "dy_lep_m50_ht800to1200_madgraph",
    "dy_lep_m50_ht1200to2500_madgraph",
    "dy_lep_m50_ht2500_madgraph",
    # WJets
    "w_lnu_ht70To100_madgraph",
    "w_lnu_ht100To200_madgraph",
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
    "qcd_ht50to100_madgraph",
    "qcd_ht100to200_madgraph",
    "qcd_ht200to300_madgraph",
    "qcd_ht300to500_madgraph",
    "qcd_ht500to700_madgraph",
    "qcd_ht700to1000_madgraph",
    "qcd_ht1000to1500_madgraph",
    "qcd_ht1500to2000_madgraph",
    "qcd_ht2000_madgraph",
]
for dataset_name in dataset_names:
    # add the dataset
    dataset = cfg.add_dataset(campaign.get_dataset(dataset_name))

    # mark the presence of a top quark
    if any(dataset_name.startswith(s) for s in ("tt", "st")):
        dataset.add_tag("has_top")

    # for testing purposes, limit the number of files to 1
    for info in dataset.info.values():
        info.n_files = min(info.n_files, 1)

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
        "Muon.pfRelIso03_all", "Muon.pfRelIso04_all",
        "Muon.jetIdx", "Muon.nStations",

        # electrons
        "Electron.pt", "Electron.eta", "Electron.phi", "Electron.mass",
        "Electron.pdgId",
        "Electron.deltaEtaSC",
        "Electron.pfRelIso03_all",

        # AK4 jets
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        "Jet.btagDeepFlavB", "Jet.hadronFlavour",
        "Jet.nConstituents", "Jet.nElectrons", "Jet.nMuons",

        # AK8 jets (TODO: optimize)
        "FatJet.*",
        #"FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass",
        #"FatJet.msoftdrop", "FatJet.tau1", "FatJet.tau2", "FatJet.tau3",
        #"FatJet.subJetIdx1", "FatJet.subJetIdx2",

        # subjets
        "SubJet.btagDeepB",

        # generator quantities (TODO: optimize)
        "Generator.*",

        # generator particles
        "GenPart.*",

        # missing transverse momentum
        "MET.pt", "MET.phi", "MET.significance", "MET.covXX", "MET.covXY", "MET.covYY",

        # number of primary vertices
        "PV.npvs",

        #
        # columns added during selection
        #

        # generator top-quark decay info
        "GenTopDecay.*",

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
})

# named references to actual versions to use for certain sets of tasks
main_ver = "v2"
cfg.x.named_versions = DotDict.wrap({
    "default": f"{main_ver}",
    "calibrate": f"{main_ver}",
    "select": f"{main_ver}",
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
        # "subjet_btag": "btagDeepCSV",  # TODO: implement
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
        "btag_wp": "medium",
    },
})

# TODO: implement
# https://twiki.cern.ch/twiki/bin/view/CMS/JetTopTagging?rev=41
cfg.x.jet_selection.ak8_subjetBtag = DotDict.wrap(cfg.x.jet_selection.ak8, **{
    # "subjet_btag": "btagDeepCSV",
    # "subjet_btag_wp": cfg.x.btag_working_points.deepcsv.loose,
    "subjet_btag" "btagDeepB"
    "subjet_btag_wp": cfg.x.btag_working_points.deepjet.loose,
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
