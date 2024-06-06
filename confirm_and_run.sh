# Function to prompt user for confirmation if command is supposed to be skipped
# if user types n or no then the command is executed
# if user types anything else or nothing then the command is skipped
# it is also possible to extend the command afte typing n or no
confirm_and_run() {
    cmd="$1"
    echo -e "${YELLOW}$cmd${NC}"
    echo -e -n "    ${RED}Skip?${NC} (y/n): "
    read response
    case "$response" in
        [nN][oO]|[nN])
            echo -e "    ${RED}Any additional parameters?${NC}"
            read -e -i "    " extra_params
            eval "$cmd $extra_params"
            ;;
        *)
            echo -e "    ${GREEN}Skipped!${NC}"
            ;;
    esac
}
