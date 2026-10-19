import argparse
import csv
import re
import statistics
from pathlib import Path

DIRECTIONS = [
    ("A->D", "amazon", "dslr"),
    ("A->W", "amazon", "webcam"),
    ("D->A", "dslr", "amazon"),
    ("D->W", "dslr", "webcam"),
    ("W->A", "webcam", "amazon"),
    ("W->D", "webcam", "dslr"),
]


def read_endpoint(path):
    text = path.read_text(errors="replace")
    values = [
        float(value)
        for value in re.findall(
            r"Accuracy of the network on TARGET:\s*([0-9.]+)%",
            text,
        )
    ]

    if len(values) != 15:
        raise RuntimeError(f"Expected 15 evaluations, found {len(values)}: {path}")

    if "Training time" not in text:
        raise RuntimeError(f"Training time marker missing: {path}")

    return values[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in range(3):
        for direction, source, target in DIRECTIONS:
            values = {}

            for adapter in ("gate", "scan"):
                log = (
                    args.results_root
                    / adapter
                    / f"seed{seed}"
                    / "office"
                    / f"{source}_{target}"
                    / "log_rank0.txt"
                )
                values[adapter] = read_endpoint(log)

            rows.append(
                (
                    seed,
                    direction,
                    values["gate"],
                    values["scan"],
                    values["scan"] - values["gate"],
                )
            )

    paired = args.output / "paired_results.tsv"
    with paired.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "seed",
            "direction",
            "gate",
            "scan",
            "scan_minus_gate",
        ])
        writer.writerows(rows)

    gate_all = [row[2] for row in rows]
    scan_all = [row[3] for row in rows]
    deltas = [row[4] for row in rows]

    lines = [
        "provenance=measured",
        "selection=fixed_final_endpoint",
        "complete_tasks=36/36",
        f"gate_mean={statistics.mean(gate_all):.4f}",
        f"scan_mean={statistics.mean(scan_all):.4f}",
        f"scan_minus_gate={statistics.mean(deltas):+.4f}",
        f"paired_delta_sd={statistics.stdev(deltas):.4f}",
        f"positive_pairs={sum(value > 0 for value in deltas)}/18",
    ]

    seed_deltas = []
    for seed in range(3):
        subset = [row for row in rows if row[0] == seed]
        delta = statistics.mean(row[4] for row in subset)
        seed_deltas.append(delta)
        lines.append(f"seed_{seed}_delta={delta:+.4f}")

    for direction, _, _ in DIRECTIONS:
        subset = [row for row in rows if row[1] == direction]
        delta = statistics.mean(row[4] for row in subset)
        lines.append(f"direction_{direction}_delta={delta:+.4f}")

    lines.extend([
        f"positive_seed_means={sum(value > 0 for value in seed_deltas)}/3",
        "interpretation=Gate and Scan provide comparable Office-31 performance.",
    ])

    summary = "\n".join(lines) + "\n"
    (args.output / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
