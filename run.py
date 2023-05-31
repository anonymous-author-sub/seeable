import logging
import os
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd, instantiate, to_absolute_path
from loguru import logger
from omegaconf import DictConfig, MissingMandatoryValue, OmegaConf, open_dict

# from src.runner import Evaluate

log = logging.getLogger(__name__)


# create wrapper that wrap print with *
def print_wrap(*args, **kwargs):
    print("*" * 80)
    print(*args, **kwargs)
    print("*" * 80)


def hydra_print_config(cfg: DictConfig):
    print_wrap(OmegaConf.to_yaml(cfg))


def hydra_dir_debug():
    print("-" * 80)
    print(f"Working directory : {os.getcwd()}")
    print(f"Orig working directory    : {get_original_cwd()}")
    print(f"to_absolute_path('foo')   : {to_absolute_path('foo')}")
    print(f"to_absolute_path('/foo')  : {to_absolute_path('/foo')}")
    log.info("Info level message")
    log.debug("Debug level message")
    print("-" * 80)


def hydra_save_config(cfg: DictConfig, name: str = "config.yaml"):
    """Save the config to a file"""
    # root = Path(cfg.root.get("root", "output"))
    # root.mkdir(exist_ok=True, parents=True)
    root = Path.cwd()
    with open(root / name, "w") as f:
        # f.write(OmegaConf.to_yaml(cfg))
        OmegaConf.save(config=cfg, f=f)


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    logger.debug(f"hydra version: {hydra.__version__}")
    hydra_print_config(cfg)
    if 1:
        hydra_save_config(cfg)
        hydra_dir_debug()

    runner = instantiate(cfg.runner, cfg)

    # print("runner.root", getattr(runner, "root", None))

    if cfg.get("confirm", False):
        input("Press any keys to continue ... \n")
    print("*" * 80)

    runner.run()


if __name__ == "__main__":
    main()
