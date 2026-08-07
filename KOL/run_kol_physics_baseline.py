from __future__ import annotations

import hydra
from psp_helper.config import MainConfig

from KOL.common.operator_export import export_operator_features


@hydra.main(
    version_base=None,
    config_path="../third_party/dl_fault_repo/config",
    config_name="main-config.yaml",
)
def main(config: MainConfig) -> None:
    run(config)


def run(config: MainConfig):
    """Callable entry point retained alongside the archival Hydra CLI."""
    return export_operator_features(config)


if __name__ == "__main__":
    main()
