from __future__ import annotations

import logging

import hydra
from psp_helper.config import MainConfig
from psp_helper.utils.logging import get_logger

from KOL.training.kol_experiment import run_kol_cv_experiment


logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../third_party/dl_fault_repo/config",
    config_name="main-config.yaml",
)
def main(config: MainConfig) -> None:
    run(config)


def run(config: MainConfig):
    """Callable entry point retained alongside the archival Hydra CLI."""
    return run_kol_cv_experiment(config=config, logger=logger)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    main()
