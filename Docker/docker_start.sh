#!/usr/bin/env bash

SCRIPT=$(realpath "$0")
SCRIPTDIR=$(dirname $SCRIPT)

sed -i 's/\r$//' "$SCRIPTDIR/set_python_env.sh"


docker build $SCRIPTDIR -f $SCRIPTDIR/Dockerfile -t my_python_env:3.10.11

WORKSPACE=$(dirname $(dirname "$SCRIPT"))

docker run -v /${WORKSPACE}:/YoloPy  --interactive --tty my_python_env:3.10.11