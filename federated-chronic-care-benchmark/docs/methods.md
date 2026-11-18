# Benchmark methods

## Study design

The fixture generates synthetic FHIR Patient, Condition, and Observation records at three simulated clinics. The cohort requires an existing prediabetes record, no type 2 diabetes record by the prediction date, and complete simulated follow-up through day 365. Final A1c values must have been measured and issued by the prediction date. The features are age, most recent A1c, and the change since the preceding distinct measurement day. The model includes missingness indicators for the two A1c features.

North and south 2022 records are used for training. Different north and south patients from 2024 form the later-period tests. Coastal 2024 records are reserved for transfer evaluation. Their simulation has a higher event probability and more missing historical A1c. The coastal comparison combines a time shift with a site shift. Because coastal has no training cohort, only pooled and federated models are evaluated there.

## Model comparison

All arms use logistic regression with the same fixed feature transformations, starting coefficients, learning rate, and L2 penalty on non-intercept terms. Fixed feature transformations avoid a separate cross-site statistics exchange. The values are simulation design constants.

- Local training applies 400 full-batch gradient steps to each training clinic independently.
- Pooled training applies 400 full-batch steps to the combined training rows.
- FedAvg broadcasts a shared starting model in each of 80 rounds. Each training clinic takes five full-batch steps on its own rows and returns its updated coefficients plus its training count. The aggregator takes a training-count-weighted average.

These settings match gradient step counts per training trajectory but do not imply equal compute or network costs. The benchmark reports AUROC, average precision, Brier score, event counts, and calibration bins for each evaluation clinic and method.

## Privacy and interpretation

Training and aggregation are simulated in a single Python process. The simulated FedAvg aggregator receives individual clinic coefficients and counts. Evaluation also runs centrally on the synthetic fixtures. Model updates and counts can still disclose information about training data. Secure aggregation, differential privacy, network transport, authentication, and formal privacy evaluation are outside this release. The benchmark is an implementation and evaluation example for distributed training, with no clinical validation.
