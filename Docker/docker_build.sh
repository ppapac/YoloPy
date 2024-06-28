#!/usr/bin/env bash

sed -i 's/\r$//' "Docker/set_python_env.sh"

docker build Docker -f Docker/Dockerfile -t my_python_env:3.10.11