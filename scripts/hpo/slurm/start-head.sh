#!/bin/bash

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "starting ray head node"
# Launch the head node
ray start --head --node-ip-address=$1 --port=6379 --redis-password=$2 --temp-dir="/home/chrispr/raylogs"
sleep infinity
