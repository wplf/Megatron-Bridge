#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3 30B-A3B MoE Pretraining Script with optional YAML and CLI overrides.

This script loads the base configuration from the Qwen3 30B-A3B recipe and allows
you to override fields using a YAML file and/or Hydra-style CLI dotlist overrides.

Examples:
    Basic usage with defaults:
        $ torchrun --nproc_per_node=8 pretrain_qwen3_30b_a3b.py

    With CLI overrides only:
        $ uv run python -m torch.distributed.run --nproc_per_node=8 examples/recipes/qwen/pretrain_qwen3_30b_a3b.py \
            model.tensor_model_parallel_size=4 \
            model.pipeline_model_parallel_size=2 \
            train.global_batch_size=64 \
            train.micro_batch_size=1

    With a YAML config file plus CLI overrides (CLI takes precedence):
        $ torchrun --nproc_per_node=8 pretrain_qwen3_30b_a3b.py \
            --config-file my_overrides.yaml optimizer.lr=0.0002
"""

import argparse
import logging
import os
import sys
from typing import Tuple

from omegaconf import OmegaConf

from megatron.bridge.recipes.qwen import qwen3_30b_a3b_pretrain_config as pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.utils.omegaconf_utils import (
    apply_overrides,
    create_omegaconf_dict_config,
    parse_hydra_overrides,
)
from megatron.bridge.utils.common_utils import get_rank_safe


logger: logging.Logger = logging.getLogger(__name__)


def parse_cli_args() -> Tuple[argparse.Namespace, list[str]]:
    """Parse known script args and return remaining as Hydra-style overrides."""
    parser = argparse.ArgumentParser(
        description="Pretrain Qwen3 30B-A3B MoE using Megatron-Bridge with optional YAML and CLI overrides",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Path to an OmegaConf YAML overrides file (optional).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args, cli_dotlist_overrides = parser.parse_known_args()
    return args, cli_dotlist_overrides


def main() -> None:
    """
    Load the base recipe config, apply YAML and CLI overrides, and start pretraining.
    """
    args, cli_overrides = parse_cli_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Megatron-Bridge Qwen3 30B-A3B MoE Pretraining Script")
    logger.info("----------------------------------------------------")

    # Load base configuration from the recipe as a Python dataclass
    cfg: ConfigContainer = pretrain_config()
    logger.info("Loaded base configuration from recipe: megatron.bridge.recipes.qwen.qwen3_30b_a3b.pretrain_config")

    # Print configuration on rank 0
    if get_rank_safe() == 0:
        cfg.print_yaml()

    # Convert to OmegaConf for merging
    merged_omega_conf, excluded_fields = create_omegaconf_dict_config(cfg)

    # Merge YAML overrides if provided
    if args.config_file:
        if not os.path.exists(args.config_file):
            logger.error(f"Override YAML file not found: {args.config_file}")
            sys.exit(1)
        yaml_overrides = OmegaConf.load(args.config_file)
        merged_omega_conf = OmegaConf.merge(merged_omega_conf, yaml_overrides)
        logger.debug("YAML overrides merged successfully.")

    # Apply CLI overrides (Hydra-style dotlist)
    if cli_overrides:
        logger.debug(f"Applying CLI overrides: {cli_overrides}")
        merged_omega_conf = parse_hydra_overrides(merged_omega_conf, cli_overrides)
        logger.debug("CLI overrides applied successfully.")

    # Apply final merged config back to the original dataclass
    final_overrides_dict = OmegaConf.to_container(merged_omega_conf, resolve=True)
    apply_overrides(cfg, final_overrides_dict, excluded_fields)

    # Display final configuration
    if get_rank_safe() == 0:
        logger.info("--- Final Merged Configuration ---")
        cfg.print_yaml()
        logger.info("----------------------------------")

    # Start pretraining
    logger.info("Starting pretraining...")
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()


