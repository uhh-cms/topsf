#
# set some common variables
# (for sourcing inside derived scripts)
#

# version tag
export my_version="test"

# background process & dataset
#   - standard model ttbar (semileptonic decay channel)
export my_process="tt"
export my_dataset="tt_sl_powheg"

export all_processes="tt,st,dy_lep,w_lnu,vv,qcd"
export all_selector_steps="LeptonTrigger,Lepton,AddLeptonVeto,MET,WLepPt,FatJet,Jet,JetLepton2DCut,BJetLeptonDeltaR,METFilters"

# categories
export my_categories="1e,1m"

# print or run commands depending on env var PRINT
_law=$(type -fp law)
law () {
    if [ -z $PRINT ]; then
        ${_law} "$@"
    else
        echo law "$@"
    fi
}

_mtt_inspect=$(type -fp mtt_inspect)
mtt_inspect () {
    if [ -z $PRINT ]; then
        ${_mtt_inspect} "$@"
    else
        echo mtt_inspect "$@"
    fi
}
