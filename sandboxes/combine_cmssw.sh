#!/usr/bin/env bash

# Script that sets up a standalone combine environment in $CF_COMBINE_BASE.
# For more info on functionality and parameters, see the generic setup script _setup_combine.sh.

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    # set variables and source the combine setup
    export CF_SANDBOX_FILE="${CF_SANDBOX_FILE:-${this_file}}"
    export CF_COMBINE_GIT_URL="${CF_COMBINE_GIT_URL:-https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git}"
    export CF_COMBINE_SCRAM_ARCH="el9_amd64_gcc12"
    export CF_COMBINE_CMSSW_VERSION="CMSSW_14_1_0_pre5"  # from combine docu 23.07.24
    export CF_COMBINE_VERSION="${CF_COMBINE_VERSION:-v10.0.1}"  # from combine docu 23.07.24
    export CF_COMBINE_ENV_NAME="$( basename "${this_file%.sh}" )"
    export CF_COMBINE_FLAG="1"  # increment when content changed

    source "${this_dir}/_setup_combine.sh" cmssw "$@"
}
action "$@"
