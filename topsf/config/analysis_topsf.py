# coding: utf-8

"""
Configuration of the topsf analysis.
"""

import law
import order as od
import os


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
# set up configs
#

from topsf.config.config_run2 import add_config as add_config_run2
from cmsdb.campaigns.run2_2017_nano_v9 import campaign_run2_2017_nano_v9

# default config
config_2017 = add_config_run2(
    analysis_topsf,
    campaign_run2_2017_nano_v9.copy(),
    config_name="campaign_run2_2017_nano_v9",
    config_id=1701,
)

# config with limited number of files
config_2017_limited = add_config_run2(
    analysis_topsf,
    campaign_run2_2017_nano_v9.copy(),
    config_name="campaign_run2_2017_nano_v9_limited",
    config_id=101701,
    limit_dataset_files=1,
)
