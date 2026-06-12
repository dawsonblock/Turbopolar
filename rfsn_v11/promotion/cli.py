"""Command-line interface for evaluating promotion evidence."""

import argparse
import json
from pathlib import Path

from rfsn_v11.promotion.gate import PromotionGate
from rfsn_v11.promotion.schema import PromotionEvidence


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TurboPolar promotion evidence."
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Path to promotion_evidence.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write promotion_decision.json (default: print to stdout)",
    )
    args = parser.parse_args()

    data = json.loads(args.evidence.read_text())
    evidence = PromotionEvidence.from_dict(data)
    decision = PromotionGate().evaluate(evidence)

    decision_dict = {
        "state": decision.state.value,
        "reasons": decision.reasons,
    }

    if args.output:
        args.output.write_text(json.dumps(decision_dict, indent=2))
        print(f"Decision written to {args.output}")
    else:
        print(json.dumps(decision_dict, indent=2))


if __name__ == "__main__":
    main()
