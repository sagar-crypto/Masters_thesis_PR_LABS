from omegaconf import OmegaConf
from .common import compose_config, parser

def main() -> int:
    args = parser("Print a fully resolved canonical experiment configuration").parse_args()
    print(OmegaConf.to_yaml(compose_config(args.overrides), resolve=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())

