import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract a fixed final endpoint from a CFA-Mamba log."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-evaluations", type=int, required=True)
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    values = [
        float(value)
        for value in re.findall(
            r"Accuracy of the network on TARGET:\s*([0-9.]+)%", text
        )
    ]

    if len(values) != args.expected_evaluations:
        raise SystemExit(
            f"Expected {args.expected_evaluations} evaluations, "
            f"found {len(values)}"
        )

    if "Training time" not in text:
        raise SystemExit("Training time marker is missing")

    print(f"endpoint_accuracy={values[-1]:.4f}")
    print(f"evaluations={len(values)}")
    print("selection=fixed_final_endpoint")


if __name__ == "__main__":
    main()
