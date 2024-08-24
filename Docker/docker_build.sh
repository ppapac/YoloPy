#!/bin/bash

SCRIPT=$(realpath "$0")
SCRIPTDIR=$(dirname $SCRIPT)
my_container="my_container"

docker build $SCRIPTDIR -f $SCRIPTDIR/Dockerfile -t my_python_env:3.10.11

WORKSPACE=$(dirname $(dirname "$SCRIPT"))

docker run --name $my_container -v /${WORKSPACE}:/YoloPy  --interactive --tty my_python_env:3.10.11
