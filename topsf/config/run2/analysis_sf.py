# coding: utf-8

"""
Configuration of the Run2 top-tagging scale factor analysis.

This analysis is meant for deriving data-to-simulation scale factors
for the cut-based top-tagging of large-radius jets based on
substructure variables.
"""

import law
import order as od
import os


thisdir = os.path.dirname(os.path.abspath(__file__))

#
# the main analysis object
#

analysis_sf = ana = od.Analysis(
    name="analysis_run2_sf",
    id=2_01_00_00,
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

from topsf.config.run2.config_sf import add_config
from cmsdb.campaigns.run2_2017_nano_v9 import campaign_run2_2017_nano_v9

# default config
config_2017 = add_config(
    analysis_sf,
    campaign_run2_2017_nano_v9.copy(),
    config_name="run2_sf_2017_nano_v9",
    config_id=2_01_17_01,
)

# config with limited number of files
config_2017_limited = add_config(
    analysis_sf,
    campaign_run2_2017_nano_v9.copy(),
    config_name="run2_sf_2017_nano_v9_limited",
    config_id=2_01_17_02,
    limit_dataset_files=1,
)
