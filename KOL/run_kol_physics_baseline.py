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
    export_operator_features(config)


if __name__ == "__main__":
    main()
