"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os

from ocpmodels.common.registry import registry
from ocpmodels.trainers.forces_trainer import ForcesTrainer


class BaseTask:
    def __init__(self, config):
        self.config = config

    def setup(self, trainer):
        self.trainer = trainer
        if self.config["checkpoint"] is not None:
            self.trainer.load_checkpoint(self.config["checkpoint"], self.config.get("strict_load", True))

        # save checkpoint path to runner state for slurm resubmissions
        self.chkpt_path = os.path.join(
            self.trainer.config["cmd"]["checkpoint_dir"], "checkpoint.pt"
        )

    def run(self):
        raise NotImplementedError


@registry.register_task("train")
class TrainTask(BaseTask):
    def run(self):
        self.trainer.train(
            disable_eval_tqdm=self.config.get("hide_eval_progressbar", False)
        )


@registry.register_task("predict")
class PredictTask(BaseTask):
    def run(self):
        assert (
            self.trainer.test_loader is not None
        ), "Test dataset is required for making predictions"
        assert self.config["checkpoint"]
        results_file = "predictions"
        self.trainer.predict(
            self.trainer.test_loader,
            results_file=results_file,
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("validate")
class ValidateTask(BaseTask):
    def run(self):
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )


@registry.register_task("run-relaxations")
class RelxationTask(BaseTask):
    def run(self):
        assert isinstance(
            self.trainer, ForcesTrainer
        ), "Relaxations are only possible for ForcesTrainer"
        assert (
            self.trainer.relax_dataset is not None
        ), "Relax dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.run_relaxations()
