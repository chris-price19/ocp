#!/usr/bin/python

import os
import time

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
# from ray.tune.utils import wait_for_gpu

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports
from datetime import datetime

import numpy as np


def ocp_trainable(config, checkpoint_dir=None):
    # wait_for_gpu()
    setup_imports()
    # trainer defaults are changed to run HPO
    trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
        task=config["task"],
        model=config["model"],
        dataset=config["dataset"],
        optimizer=config["optim"],
        identifier=config["identifier"],
        run_dir=config.get("run_dir", "./"),
        is_debug=config.get("is_debug", False),
        is_vis=config.get("is_vis", False),
        is_hpo=config.get("is_hpo", True),  # hpo
        print_every=config.get("print_every", 10),
        seed=config.get("seed", 0),
        logger=config.get("logger", None),  # hpo
        local_rank=config["local_rank"],
        amp=config.get("amp", False),
        cpu=config.get("cpu", False),
    )
    # add checkpoint here
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        trainer.load_pretrained(checkpoint)
    
    trainer.train()

# this function is general and should work for any ocp trainer
def test_trainable(config):

    # trainer defaults are changed to run HPO
    i = 0
    while i < 1000:
        i += 1
        time.sleep(3)

def main():

    # config = {"model": "abc", "run_dir": "./logs"}
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)
    
    config["task"].update(
        loss_balance=tune.choice([0.1, 1., 10.]), # cut down for full dataset
    )

    config["model"].update(
        hidden_channels=tune.choice([32, 64, 96, ]), # cut down for full dataset
        out_emb_channels=tune.choice([24, 48, 64, 96 ]), # cut down for full dataset
        num_blocks=tune.choice([2, 3,]),
        num_radial=tune.choice([4, 5, 6, ]),
        num_spherical=tune.choice([4, 5, 6,]),
        num_output_layers=tune.choice([2, 4]),
    )

    lr_milestones = [[30000, 60000, 12000],
                      [75000, 125000]]

    ## I think something like - update yes this works
    config["optim"].update(
        lr_initial=tune.choice([1e-2, 5e-3, 1e-3]),
        lr_milestones=tune.sample_from(lambda spec: lr_milestones[np.random.randint(len(lr_milestones))]),
        batch_size=tune.choice([4, 8, 12]), # cut down for full dataset
        warmup_steps=tune.choice([100, 500]),
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

    ray.init()


    datastring = config["dataset"][0]["src"].split('/')[-2].split('_')[0]
    # define run parameters
    analysis = tune.run(
        ocp_trainable,
        name=config["model"]["name"] + '-' + datastring + '-' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S"),
        resources_per_trial={"cpu": 1},
        config=config,
        fail_fast=True,
        local_dir=config.get("run_dir", "./"),
        num_samples=8,
    )


if __name__ == "__main__":
    main()
