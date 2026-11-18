"""Compare local, pooled, and FedAvg training in a synthetic clinic simulation."""
import argparse
from pathlib import Path

import numpy as np

from study_fixture import HORIZON, build_cohort, generate, metrics, write_json

ROUNDS = 80
LOCAL_STEPS = 5
LEARNING_RATE = 0.3
REGULARIZATION = 0.002


def design(rows):
    """Use fixed reference values so preprocessing requires no cross-site fit."""
    features = []
    for row in rows:
        latest = row["latest_a1c"]
        change = row["a1c_change"]
        features.append([
            1.0,
            (row["age_years"] - 55) / 15,
            (latest - 6) / 0.5 if latest is not None else 0.0,
            change / 0.25 if change is not None else 0.0,
            float(latest is None),
            float(change is None),
        ])
    return np.asarray(features, dtype=float), np.asarray([row["outcome"] for row in rows])


def predict(features, weights):
    return 1 / (1 + np.exp(-np.clip(features @ weights, -50, 50)))


def update(weights, data, steps):
    features, labels = data
    weights = weights.copy()
    for _ in range(steps):
        gradient = features.T @ (predict(features, weights) - labels) / len(labels)
        gradient[1:] += REGULARIZATION * weights[1:]
        weights -= LEARNING_RATE * gradient
    return weights


def aggregate(updates):
    """Weight each clinic's updated coefficients by its training cohort size."""
    weights, counts = zip(*updates)
    return np.average(np.stack(weights), axis=0, weights=counts)


def federate(clinics, initial, rounds=ROUNDS, local_steps=LOCAL_STEPS):
    weights = initial.copy()
    for _ in range(rounds):
        updates = [
            (update(weights, data, local_steps), len(data[1]))
            for data in clinics.values()
        ]
        weights = aggregate(updates)
    return weights


def run(seed):
    bundle, metadata = generate(seed)
    rows = build_cohort(bundle, metadata)
    training = [row for row in rows if row["split"] == "train"]
    clinic_ids = sorted({row["site"] for row in training})
    clinics = {
        site: design([row for row in training if row["site"] == site])
        for site in clinic_ids
    }
    initial = np.zeros(next(iter(clinics.values()))[0].shape[1])
    iterations = ROUNDS * LOCAL_STEPS
    local = {site: update(initial, data, iterations) for site, data in clinics.items()}
    pooled = update(initial, design(training), iterations)
    federated = federate(clinics, initial)

    evaluation = {}
    for site in sorted({row["site"] for row in rows if row["split"] != "train"}):
        cohort = [row for row in rows if row["site"] == site and row["split"] != "train"]
        features, _ = design(cohort)
        candidates = {"pooled": pooled, "fedavg": federated}
        if site in local:
            candidates["site_local"] = local[site]
        evaluation[site] = {
            "split": cohort[0]["split"],
            "models": {
                name: metrics(cohort, predict(features, weights))
                for name, weights in candidates.items()
            },
        }
    return {
        "synthetic_only": True,
        "seed": seed,
        "horizon_days": HORIZON,
        "training_counts": {site: len(data[1]) for site, data in clinics.items()},
        "rounds": ROUNDS,
        "local_steps_per_round": LOCAL_STEPS,
        "learning_rate": LEARNING_RATE,
        "l2_strength_excluding_intercept": REGULARIZATION,
        "evaluation": evaluation,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("build/benchmark.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, run(args.seed))
    print(f"Benchmark report: {args.out}")
