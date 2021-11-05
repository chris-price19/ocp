import os

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.utils import wait_for_gpu

from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import build_config, setup_imports
from datetime import datetime

# os.environ['http_proxy'] = ''
# os.environ['https_proxy'] = ''

# this function is general and should work for any ocp trainer
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
    # start training
    print('traintype', flush=True)
    print(type(trainer), flush=True)

    trainer.train()


# this section defines the hyperparameters to tune and all the Ray Tune settings
# current params/settings are an example for ForceNet
def main():
    # parse config
    parser = flags.get_parser()
    args, override_args = parser.parse_known_args()
    config = build_config(args, override_args)

    # add parameters to tune using grid or random search
    # config["model"].update(
    #     hidden_channels=tune.choice([256, 384, 512, 640, 704]),
    #     decoder_hidden_channels=tune.choice([256, 384, 512, 640, 704]),
    #     depth_mlp_edge=tune.choice([1, 2, 3, 4, 5]),
    #     depth_mlp_node=tune.choice([1, 2, 3, 4, 5]),
    #     num_interactions=tune.choice([3, 4, 5, 6]),
    # )
    ## dpp - what about optimizer params? can anything in config.yml go here?
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


    # ray init
    # for debug
    # ray.init(local_mode=True,num_cpus=32, num_gpus=8,  _temp_dir="/home/chrispr/raylogs")
    # for cluster interactive session
    # ray.init(address="auto", _temp_dir="/home/chrispr/raylogs", _redis_password='4fe577fb-1673-4760-9907-f46ef54a013a')

    # for slurm cluster
    ray.init(
        address="auto",
        _node_ip_address=os.environ["ip_head"].split(":")[0],
        _redis_password=os.environ["redis_password"],
        _temp_dir="/home/chrispr/raylogs",
    )
    # define command line reporter
    reporter = CLIReporter(
        print_intermediate_tables=True,
        metric="val_loss",
        mode="min",
        metric_columns={
            "steps": "steps",
            "epochs": "epochs",
            "training_iteration": "training_iteration",
            "val_loss": "val_loss",
            "val_energy_mae": "val_energy_mae",
        },
    )

    print(config.get("run_dir", "./"))
    # define run parameters
    analysis = tune.run(
        ocp_trainable,
        name=config["model"]["name"] + '-' + datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S"),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=config,
        fail_fast=True,
        local_dir=config.get("run_dir", "./"),
        num_samples=512,
        progress_reporter=reporter,
        scheduler=scheduler,

    )

    print(
        "Best config is:",
        analysis.get_best_config(
            metric="val_energy_mae", mode="min", scope="last"
        ),
    )


if __name__ == "__main__":
    main()
