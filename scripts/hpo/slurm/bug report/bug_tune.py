#!/usr/bin/python

import os
import time

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.utils import wait_for_gpu

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# this function is general and should work for any ocp trainer
def test_trainable(config):

    # trainer defaults are changed to run HPO
    i = 0
    while i < 1000:
        i += 1
        time.sleep(3)

def main():

    config = {"model": "abc", "run_dir": "./logs"}
    ## dpp - what about optimizer params? can anything in config.yml go here?
    # config["model"].update(
    #     hidden_channels=tune.choice([32, 64, 128]),
    #     out_emb_channels=tune.choice([24, 48, 96]),
    #     num_blocks=tune.choice([1, 2, 3]),
    #     num_radial=tune.choice([4, 5, 6]),
    #     num_spherical=tune.choice([4, 5, 6]),
    #     num_output_layers=tune.choice([1,2,3]),
    # )

    # for slurm cluster
    ray.init(
        address="auto",
        _node_ip_address=os.environ["ip_head"].split(":")[0],
        _redis_password=os.environ["redis_password"],
        _temp_dir="/home/chrispr/raylogs",
    )

    # define run parameters
    analysis = tune.run(
        test_trainable,
        resources_per_trial={"cpu": 4},
        config=config,
        fail_fast=True,
        local_dir=config.get("run_dir", "./"),
        num_samples=64,
    )


if __name__ == "__main__":
    main()
