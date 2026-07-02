# Paper Alignment

This note maps the ICML paper logic to the cleaned implementation.

## Algorithm 1: Gradient-Based Data Cleaning Sampling

Paper:

- Candidate differentiable models come from AutoML.
- For each sample, compute the gradient norm under each model.
- Aggregate by mean plus variance: `score_i = mean(G_i) + alpha * var(G_i)`.
- Clean the top-k scored samples.

Code:

- `dmco.gradients.aggregate_gradient_scores`
- `dmco.sampling.gradient_cleaning_sample`
- `dmco.data.apply_reference_cleaning`
- Called by `DMCORunner._clean_step`

The old scripts used fixed fractions and sometimes a single SVR/SVC proxy. The new version uses the current AutoML candidate family and keeps an explicit `cleaned_mask` so rows are not cleaned repeatedly.

## Algorithm 2: Loss-Driven Progressive AutoML

Paper:

- Use the current best model to compute per-sample loss.
- Mix loss weights with uniform weights using beta.
- Apply Gumbel-top-k sampling without replacement.
- Reuse and prune good model families across time slices.

Code:

- `dmco.metrics.per_sample_loss`
- `dmco.sampling.loss_driven_gumbel_sample`
- `dmco.automl.AutoMLAdapter`
- Called by `DMCORunner._automl_step`

The implementation supports an `auto-sklearn` backend when installed, but defaults to a scikit-learn backend so the repository remains runnable on macOS and CI.

## Algorithm 3: Cost-Aware MAB Time-Slice Allocation

Paper:

- Partition total budget into time slices.
- Treat cleaning and AutoML as arms.
- Reward is metric improvement per unit cost.
- Use discounted UCB with stagnation-triggered exploration.

Code:

- `dmco.config.BudgetConfig.slice_seconds`
- `dmco.scheduler.CostAwareUCB`
- `dmco.pipeline.DMCORunner.fit`

The previous scripts hard-coded values such as 30/60/300 seconds and fixed one-half or one-third splits. The new version derives per-slice behavior from `configs/dmco.yaml`.

## Baselines

The package focuses on the reusable DMCO algorithm. The historical baseline scripts are kept in `legacy/` for traceability. For new experiments, use the package modules to implement DR, DA, CR, CA, and DMCO under the same configuration and data split.

## GitHub Hygiene

The curated CSV artifact under `data/raw/` is intentionally tracked so a fresh clone can run the
included reproducibility scripts. IDE metadata, caches, virtual environments, temporary logs, and
non-curated local outputs remain ignored by `.gitignore`.
