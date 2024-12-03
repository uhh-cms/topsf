# coding: utf-8

"""
++++++++++ WIP ++++++++++
Configuration of the Run3 top-tagging scale factor analysis,
based on the Run2 top-tagging scale factor analysis.

This analysis is meant for deriving data-to-simulation scale factors
for the cut-based top-tagging of large-radius jets based on
substructure variables.
++++++++++ WIP ++++++++++
"""

import law
import order as od
import os


thisdir = os.path.dirname(os.path.abspath(__file__))

#
# the main analysis object
#

analysis_sf = ana = od.Analysis(
    name="analysis_run3_sf",
    id=1_03_00_00,  # 1: SF 03: Run3 00: year 0: full stat 0: campaign
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
    "$TOPSF_BASE/sandboxes/combine_cmssw.sh",
]

# clear the list when cmssw bundling is disabled
if not law.util.flag_to_bool(os.getenv("TOPSF_BUNDLE_CMSSW", "1")):
    del ana.x.cmssw_sandboxes[:]

# config groups for conveniently looping over certain configs
# (used in wrapper_factory)
ana.x.config_groups = {}


#
# set up configs
#

from topsf.config.run3.config_sf import add_config
import cmsdb.campaigns.run3_2022_preEE_nano_v12
import cmsdb.campaigns.run3_2022_postEE_nano_v12
# import cmsdb.campaigns.run3_2023_preBPix_nano_v12
# import cmsdb.campaigns.run3_2023_postBPix_nano_v12

campaign_run3_2022_preEE_nano_v12 = cmsdb.campaigns.run3_2022_preEE_nano_v12.campaign_run3_2022_preEE_nano_v12
campaign_run3_2022_preEE_nano_v12.x.EE = "pre"

campaign_run3_2022_postEE_nano_v12 = cmsdb.campaigns.run3_2022_postEE_nano_v12.campaign_run3_2022_postEE_nano_v12
campaign_run3_2022_postEE_nano_v12.x.EE = "post"

# campaign_run3_2023_preBPix_nano_v12 = cmsdb.campaigns.run3_2023_preBPix_nano_v12.campaign_run3_2023_preBPix_nano_v12
# campaign_run3_2023_preBPix_nano_v12.x.BPix = "pre"

# campaign_run3_2023_postBPix_nano_v12 = cmsdb.campaigns.run3_2023_postBPix_nano_v12.campaign_run3_2023_postBPix_nano_v12
# campaign_run3_2023_postBPix_nano_v12.x.BPix = "post"

# default config
config_2022_preEE = add_config(
    analysis_sf,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_sf_2022_preEE_nano_v12",
    config_id=1_03_22_11,  # 1: SF 03: Run3 22: year 1: full stat 1: pre EE
)

config_2022_postEE = add_config(
    analysis_sf,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_sf_2022_postEE_nano_v12",
    config_id=1_03_22_12,  # 1: SF 03: Run3 22: year 1: full stat 2: post EE
)

# config_2023_preBPix = add_config(
#     analysis_sf,
#     campaign_run3_2023_preBPix_nano_v12.copy(),
#     config_name="run3_sf_2023_preBPix_nano_v12",
#     config_id=1_03_23_11,  # 1: SF 03: Run3 23: year 1: full stat 1: pre BPix
# )

# config_2023_postBPix = add_config(
#     analysis_sf,
#     campaign_run3_2023_postBPix_nano_v12.copy(),
#     config_name="run3_sf_2023_postBPix_nano_v12",
#     config_id=1_03_23_12,  # 1: SF 03: Run3 23: year 1: full stat 2: post BPix
# )

# config with limited number of files
config_2022_preEE_limited = add_config(
    analysis_sf,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_sf_2022_preEE_nano_v12_limited",
    config_id=1_03_22_21,  # 1: SF 03: Run3 22: year 2: limited stat 1: pre EE
    limit_dataset_files=1,
)

config_2022_postEE_limited = add_config(
    analysis_sf,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_sf_2022_postEE_nano_v12_limited",
    config_id=1_03_22_22,  # 1: SF 03: Run3 22: year 2: limited stat 2: post EE
    limit_dataset_files=1,
)

# config_2023_preBPix_limited = add_config(
#     analysis_sf,
#     campaign_run3_2023_preBPix_nano_v12.copy(),
#     config_name="run3_sf_2023_preBPix_nano_v12_limited",
#     config_id=1_03_23_21,  # 1: SF 03: Run3 23: year 2: limited stat 1: pre BPix
#     limit_dataset_files=1,
# )

# config_2023_postBPix_limited = add_config(
#     analysis_sf,
#     campaign_run3_2023_postBPix_nano_v12.copy(),
#     config_name="run3_sf_2023_postBPix_nano_v12_limited",
#     config_id=1_03_23_22,  # 1: SF 03: Run3 23: year 2: limited stat 2: post BPix
#     limit_dataset_files=1,
# )
