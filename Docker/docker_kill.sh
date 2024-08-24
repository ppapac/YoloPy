#!/bin/bash

my_container="my_container"

containers=$(docker ps -a -f name=$my_container | wc -l)
containers_running=$(docker ps -f name=$my_container | wc -l)

if [ $containers_running -eq 2 ]; then
    echo "Running containers found. Proceeding to stop it."
    docker kill $my_container
else
    echo "No running container $my_container found."
fi

if [ $containers -eq 2 ]; then
    echo "Deleting container..."
    docker rm $my_container
else
    echo "No container $my_container found."
fi