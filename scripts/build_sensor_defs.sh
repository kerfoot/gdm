#! /bin/bash --

PATH=${PATH}:/bin;

app=$(basename $0);
app_dir=$(realpath $(dirname $0));

defs_root=$(realpath "${app_dir}/../resources/defs");
allowed_def_types='gdm base cf derived renamed ngdac';
logging_routines="${app_dir}/logging.sh";

# Usage message
USAGE="
NAME
    $app - build a sensor definitions file

SYNOPSIS
    $app [h]

DESCRIPTION
    -h
        show help message
";

# Default values for options

# Process options
while getopts "hd:" option
do
    case "$option" in
        "h")
            echo -e "$USAGE";
            exit 0;
            ;;
        "d")
            defs_root=$OPTARG;
            ;;
        "?")
            echo -e "$USAGE" >&2;
            exit 1;
            ;;
    esac
done

# Remove option from $@
shift $((OPTIND-1));

if [ ! -f "$logging_routines" ]
then
    echo "Logging routines not found: $logging_routines" >&2;
    exit 1;
fi

. $logging_routines;

info_msg "Logging enabled";

if [ ! -d "$defs_root" ]
then
    error_msg "Sensor definitions root not found: $defs_root";
    exit 1;
fi

info_msg "Sensor definitions root: $defs_root";

# Make a temporary directory for writing the intermediate dba files and
# (optionally) doing the dba->matlab conversions
info_msg 'Creating temporary location for individual sensor definition files...';
tmp_defs_path=$(mktemp -d -t ${app}.XXXXXXXXXX);
if [ "$?" -ne 0 ]
then
    error_msg 'Failed create temporary defs directory';
    exit 1;
fi
info_msg "Temporary defs path: $tmp_defs_path";
# Remove $tmp_defs_path if SIG
trap "{ rm -Rf $tmp_defs_path; exit 255; }" SIGHUP SIGINT SIGKILL SIGTERM SIGSTOP;

def_types="${allowed_def_types}";
[ "$#" -gt 0 ] && def_types="$@";

for def_type in $def_types
do
    defs_path="${defs_root}/${def_type}";
    if [ ! -d "$defs_path" ]
    then
        warn_msg "Definitions path does not exist: $defs_path";
        continue;
    fi

    info_msg "Checking definitions in $defs_path";

    find $defs_path -type f -name '*.yml' -exec cp '{}' $tmp_defs_path \;
done

find $tmp_defs_path -name '*.yml' -exec cat '{}' \;

info_msg "Removing temporary defs path: $tmp_defs_path";
rm -Rf $tmp_defs_path;

