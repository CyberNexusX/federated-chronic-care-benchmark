# Federated Chronic Care Benchmark

A MedScope Analytics research prototype comparing clinic-local, pooled, and federated training for one chronic-disease risk endpoint. All records are simulated. The endpoint is a **first recorded type 2 diabetes diagnosis within 365 days** among adults with recorded prediabetes.

The benchmark reuses the synthetic FHIR R4 data generator and cohort rules from the companion project, `cardiometabolic-risk-toolkit`. It runs independently with the fixture code in this repository.

## Run the benchmark

Use Python 3.11 or 3.12.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python benchmark.py
python -m unittest discover -s tests -v
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The default report is written to `build/benchmark.json`. Set `--seed` and `--out` to reproduce a particular run or compare simulations.

## Sample use cases

1. **Compare training approaches at the same clinic.** Open the north or south entry in the report to compare site-local, pooled, and FedAvg scores on exactly the same later-period patients.
2. **Evaluate transfer to a new clinic.** Inspect the coastal clinic's Brier score and compare mean predicted probability with the observed event rate. The simulated coastal clinic does not contribute to training.
3. **Review the information exchanged.** Trace the FedAvg step from global coefficients to local training and size-weighted aggregation. Each simulated update provides coefficients and a training count. Review the scope of that design in the methods.

## Default run

| Evaluation clinic | Patients | Local AUROC | Pooled AUROC | FedAvg AUROC |
|---|---:|---:|---:|---:|
| North, later period | 426 | 0.648 | 0.645 | 0.645 |
| South, later period | 421 | 0.746 | 0.744 | 0.744 |
| Coastal, held-out clinic | 831 | — | 0.683 | 0.683 |

The default simulation gives pooled and federated training similar results. The training clinics use the same underlying data-generation assumptions. This experiment does not measure robustness to substantial differences between participating clinics, performance on real patients, or any formal privacy guarantee.

See the [full sample report](docs/sample-benchmark.json), [methods](docs/methods.md), and [GitHub publishing fields](docs/github-publishing.md).
