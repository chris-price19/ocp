#!/usr/bin/python

import os
import time

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.utils import wait_for_gpu


# this function is general and should work for any ocp trainer
def test_trainable(config):

    # trainer defaults are changed to run HPO
    i = 0
    while i < 1000:
        i += 1
        time.sleep(3)

def main():

    config = {"model": "abc", "run_dir": "./logs"}
    config = build_config(args, override_args)
    
    config["model"].update(
    atom_embedding_size=tune.choice([32, 64, 96, 128, 172, 256]),
    fc_feat_size=tune.choice([64, 96, 128, 172, 256]),
    num_fc_layers=tune.choice([2, 3, 4, 5]),
    num_graph_conv_layers=tune.choice([3,4,5,6]),
    num_gaussians=tune.choice([50, 80, 110, 140]),
)

    ## I think something like - update yes this works
    config["optim"].update(
        lr_initial=tune.choice([1e-2, 5e-3, 1e-3]),
        # lr_milestones=tune.choice([[350, 700, 1400], [700, 1400, 2500]]),
        batch_size=tune.choice([16, 32, 64, 128]),
        warmup_steps=tune.choice([50, 250, 500]),
    )

    # define scheduler
    scheduler = ASHAScheduler(
        time_attr="steps",
        metric="val_loss",
        mode="min",
        max_t=100000,
        grace_period=2000,
        reduction_factor=4,
        brackets=1,
    )

    # for slurm cluster
    # ray.init(
    #     address="auto",
    #     _node_ip_address=os.environ["ip_head"].split(":")[0],
    #     _redis_password=os.environ["redis_password"],
    #     _temp_dir="/home/chrispr/raylogs",
    # )

    ray.init(local_mode=True, num_cpus=8)

    # define run parameters
    analysis = tune.run(
        test_trainable,
        resources_per_trial={"cpu": 1},
        config=config,
        fail_fast=True,
        local_dir=config.get("run_dir", "./"),
        num_samples=64,
    )


if __name__ == "__main__":
    main()
