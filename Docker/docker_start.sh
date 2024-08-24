#!/bin/bash

my_container="my_container"

containers=$(docker ps -a -f name=$my_container | wc -l)

if [ $containers -eq 2 ]; then
    echo "$my_container exists"
    docker container start -a -i my_container
else
    echo "$my_container does not exist"
fi