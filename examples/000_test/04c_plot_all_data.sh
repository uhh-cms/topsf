#!/bin/sh
action () {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    source ${this_dir}/common.sh

    args=(
        --version $my_version
        --processes data,$all_processes
        --variables $all_variables
        #--variables probejet_msoftdrop_widebins
        --categories 1m,1m__pt_600_inf__tau32_000_038,1m__pt_300_400__tau32_000_038,1e,1e__pt_600_inf__tau32_000_038,1e__pt_300_400__tau32_000_038
        --shape-norm
        "$@"
    )

    law run cf.PlotVariables1D "${args[@]}"
}

action "$@"

