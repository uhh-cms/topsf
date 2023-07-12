#
# set some common variables
# (for sourcing inside derived scripts)
#

# version tag
export my_version="v4"

# background process & dataset
#   - standard model ttbar (semileptonic decay channel)
#export my_process="tt"
#export my_dataset="tt_sl_powheg"
export my_process="st"
export my_dataset="st_tchannel_t_powheg"
#export my_process="vv"
#export my_dataset="ww_pythia"

#export all_processes="tt,st,dy_lep,w_lnu,vv,qcd"
export all_processes="mj,tt_3q,tt_2q,tt_0o1q,tt_bkg,st_3q,st_2q,st_0o1q,st_bkg,vx"
export all_selector_steps="LeptonTrigger,Lepton,AddLeptonVeto,MET,WLepPt,FatJet,Jet,JetLepton2DCut,BJetLeptonDeltaR,METFilters"

export all_variables="probejet_pt,probejet_mass,probejet_msoftdrop_widebins,probejet_tau32,probejet_max_subjet_btag_score_btagDeepB"

# categories
export all_categories=$(echo 1{e,m} 1{e,m}__pt_{300_400,400_480,480_600,600_inf}__tau32_wp_{very_loose,loose,medium,tight,very_tight}_{pass,fail} 1{e,m}__tau32_wp_{very_loose,loose,medium,tight,very_tight}_{pass,fail} | tr " " ",")
export test_categories=$(echo 1{e,m} 1{e,m}__tau32_wp_{very_loose,very_tight}_{pass,fail} | tr " " ",")
export my_categories="1m,1m__tau32_wp_very_loose_pass,1m__tau32_wp_very_loose_fail,1m__pt_600_inf__tau32_wp_very_loose_pass,1m__pt_600_inf__tau32_wp_very_loose_fail"

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
