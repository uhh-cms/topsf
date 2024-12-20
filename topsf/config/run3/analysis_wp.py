# coding: utf-8

"""
Configuration of the Run3 top-tagging working point analysis.

This analysis is meant for measuring the working points
for cut-based tau32 top tagging from a single set of
QCD MC samples.
"""

import law
import order as od
import os


thisdir = os.path.dirname(os.path.abspath(__file__))

#
# the main analysis object
#

analysis_wp = ana = od.Analysis(
    name="analysis_run3_wp",
    id=2_03_00_00,
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
# set up configs
#

from topsf.config.run3.config_wp import add_config
import cmsdb.campaigns.run3_2022_preEE_nano_v12
import cmsdb.campaigns.run3_2022_postEE_nano_v12

campaign_run3_2022_preEE_nano_v12 = cmsdb.campaigns.run3_2022_preEE_nano_v12.campaign_run3_2022_preEE_nano_v12
campaign_run3_2022_preEE_nano_v12.x.EE = "pre"

campaign_run3_2022_postEE_nano_v12 = cmsdb.campaigns.run3_2022_postEE_nano_v12.campaign_run3_2022_postEE_nano_v12
campaign_run3_2022_postEE_nano_v12.x.EE = "post"

# default config
config_2022_preEE = add_config(
    analysis_wp,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_wp_2022_preEE_nano_v12",
    config_id=2_03_22_11,
)

config_2022_postEE = add_config(
    analysis_wp,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_wp_2022_postEE_nano_v12",
    config_id=2_03_22_12,
)

# config with limited number of files
config_2022_preEE_limited = add_config(
    analysis_wp,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_wp_2022_preEE_nano_v12_limited",
    config_id=2_03_22_21,
    limit_dataset_files=1,
)

config_2022_postEE_limited = add_config(
    analysis_wp,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_wp_2022_postEE_nano_v12_limited",
    config_id=2_03_22_22,
    limit_dataset_files=1,
)

# config with limited number of files
config_2022_preEE_medium_limited = add_config(
    analysis_wp,
    campaign_run3_2022_preEE_nano_v12.copy(),
    config_name="run3_wp_2022_preEE_nano_v12_medium_limited",
    config_id=2_03_22_31,
    limit_dataset_files=10,
)

config_2022_postEE_medium_limited = add_config(
    analysis_wp,
    campaign_run3_2022_postEE_nano_v12.copy(),
    config_name="run3_wp_2022_postEE_nano_v12_medium_limited",
    config_id=2_03_22_32,
    limit_dataset_files=10,
)
