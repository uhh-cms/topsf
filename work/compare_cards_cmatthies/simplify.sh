#!/bin/bash

read_between() {
    local from=$1
    local to=$2
    local num=$(($2 - $1 + 1))
    head -${to} | tail -${num}
}

simplify() {
    sed -e "s/CombBin-Main-//g" | sed -e "s/-UL17-pt_300to400//g" | sed -e "s/__MSc//g" | sed -e "s/__UL17__pt_300to400//g"
}

columnate() {
    local file_in=$1
    local section_sep="------------------------------------------"

    [ -n "$file_in" ] || return 1

    cat "$file_in" | simplify | read_between 1 6
    cat "$file_in" | simplify | read_between 7 8 | column -t
    echo "$section_sep"
    cat "$file_in" | simplify | read_between 10 13 | column -t
    echo "$section_sep"
    cat "$file_in" | simplify | read_between 15 60 | column -t
    cat "$file_in" | simplify | read_between 61 124
}

columnate "$@"
