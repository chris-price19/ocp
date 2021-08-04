#!/bin/bash

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "starting ray worker node"
ray start --address $1 --redis-password=$2
sleep infinity
