from __future__ import annotations

import argparse

from KOL.input_builder import build_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate all four active C/L experiment input CSVs."
    )
    parser.add_argument("--raw-90-default", required=True)
    parser.add_argument("--raw-90-opposite", required=True)
    parser.add_argument("--raw-90-both", required=True)
    parser.add_argument("--raw-110-both", required=True)
    parser.add_argument(
        "--output-root", default="outputs/chapter4/model_inputs/unified_active"
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Atomically replace LATEST_ACTIVE_INPUTS.env after validation.",
    )
    args = parser.parse_args()
    output = build_inputs(
        raw_90_default=args.raw_90_default,
        raw_90_opposite=args.raw_90_opposite,
        raw_90_both=args.raw_90_both,
        raw_110_both=args.raw_110_both,
        output_root=args.output_root,
        activate=args.activate,
    )
    print(f"Published validated inputs: {output}")
    print(
        "Activated LATEST_ACTIVE_INPUTS.env"
        if args.activate
        else "Not activated (use --activate to update the runtime environment)."
    )


if __name__ == "__main__":
    main()
