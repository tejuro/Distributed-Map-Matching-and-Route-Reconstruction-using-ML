# Member 2 — ML Models & Evaluation (Part 3)

Training, comparison, Bayesian optimisation and evaluation of the candidate-arc
classifiers for PathSnap.

**Task.** Each GPS observation gets a small set of nearby candidate road arcs.
A binary classifier scores every (observation, arc) pair; the best-scoring arc
becomes that observation's predicted road. Target = `is_true_arc` (0/1). This is
never framed as a global multi-class problem over `arc_id`, because arc ids only
mean something within a single trip.

---

## Files owned by this part

| File | Purpose |
|---|---|
| `src/02_train_compare.py` | Trains and compares the five models: group CV + student-test + HMM decoding of every model |
| `src/03_bayes_opt.py` | Optuna (TPE) Bayesian optimisation, final model refit, HMM parameter tuning, final evaluation |
| `models/LogisticRegression.joblib` | Comparison model |
| `models/XGBoost.joblib` | Comparison model |
| `models/SVM.joblib` | Comparison model |
| `models/final_model.joblib` | Deployed model — Bayes-tuned XGBoost |
| `models/final_model_config.json` | Model name, feature order, tuned hyperparameters |
| `models/hmm_params.json` | HMM parameters selected on out-of-fold training probabilities |
| `results/model_comparison.csv` | Consolidated comparison table |
| `results/bayes_opt_trials.csv` | Every optimisation trial and its objective |
| `results/experiment_log.json` | Configurations and seeds for reproducibility |
| `results/final_results.json` | Headline metrics of the final system |
| `results/final_student_test_per_trip.csv` | Per-trip student-test results |
| `results/plots/*` | Comparison, confusion matrices, feature importance, Bayes history |

**Upstream dependency:** this part imports `src/features.py` (Part 2, Member 1)
and `src/hmm.py` (Part 4, Member 3). Those live on their own branches. Until
they are merged, running these scripts from this branch alone will fail on those
imports — that is expected in parallel development, not a defect.

---

## The five models

| Model | Family | Configuration |
|---|---|---|
| Logistic Regression | Linear / probabilistic | C=1.0, L2, standardised features |
| KNN | Instance-based | k=25, distance-weighted, Euclidean, standardised |
| SVM | Kernel | RBF, C=1, gamma='scale', Platt probabilities, 3k-observation subset |
| Random Forest | Ensemble (bagging) | 200 trees, min_samples_leaf=5, sqrt features |
| XGBoost | Boosting | 300 trees, lr=0.1, max_depth=6, hist |

One model per family, so selection is a real decision rather than a comparison
of near-identical ensembles. Two documented constraints:

- **SVM** is fitted on a 3,000-observation subset — a kernel SVM is quadratic in
  training rows and is intractable on the full table. This cap is reported next
  to its numbers rather than hidden.
- **KNN** inference is ~400× slower per 1,000 rows than XGBoost (837 ms vs
  2.08 ms), which disqualifies it for a live demo regardless of accuracy.

---

## Validation protocol

`GroupKFold(5)` with **groups = `trip_id`**. A trip is never split across folds,
because candidate pairs from one trip share that trip's road network and
trajectory; a row-wise split leaks information and inflates every metric.

The student-test partition is used **only** for reporting, after fitting on the
training partition. The evaluator-testing partition is never read by any code
here.

---

## Results

### Cross-validation (78 training trips)

| Model | CV Top-1 | CV macro F1 | CV log loss | Train time |
|---|---|---|---|---|
| **XGBoost** | **0.8016** | **0.8792** | **0.1422** | 2.18 s |
| Random Forest | 0.7970 | 0.8778 | 0.1420 | 41.8 s |
| Logistic Regression | 0.7892 | 0.8606 | 0.1777 | 0.65 s |
| KNN | 0.7865 | 0.8687 | 0.1919 | 0.75 s |
| SVM (RBF) | 0.7856 | 0.8608 | 0.1806 | 15.9 s |

### Student-test (10 held-out trips)

| Model | Top-1 | Macro F1 | Log loss | + Viterbi Top-1 | Route continuity |
|---|---|---|---|---|---|
| **XGBoost** | **0.8190** | 0.8907 | **0.1277** | 0.8994 | 0.9402 |
| Random Forest | 0.8163 | **0.8920** | 0.1295 | **0.8997** | **0.9520** |
| SVM (RBF) | 0.8109 | 0.8808 | 0.1657 | 0.8792 | 0.9522 |
| Logistic Regression | 0.8051 | 0.8728 | 0.1687 | 0.8683 | 0.9542 |
| KNN | 0.8002 | 0.8799 | 0.1800 | 0.8711 | 0.9437 |

### Final deployed system

| Metric | Value |
|---|---|
| Classifier Top-1 | 0.8198 |
| **+ HMM/Viterbi Top-1** | **0.9076** |
| Route continuity | 0.9803 |
| Accuracy | 0.9473 |
| Macro F1 | 0.8929 |
| Log loss | 0.1266 |

---

## Which metric decided it

**Candidate-level metrics** (precision, recall, F1, log loss) measure how well
the model scores each pair, but they are blind to whether the *winning*
candidate is the correct one.

**Observation-level Top-1 accuracy** takes the best-scoring arc per GPS point and
compares it with the true arc. This is what map-matching is actually judged on,
so it is the selection metric. Top-3 candidate recall and route continuity are
reported alongside, because a route must not only be locally right but also
connected.

XGBoost was selected on Top-1 + log loss with the best training speed. It is
also the model furthest ahead after HMM decoding (0.8994).

---

## Bayesian hyperparameter optimisation

- **Library / sampler:** Optuna with TPE (Tree-structured Parzen Estimator).
- **Objective:** mean 3-fold `GroupKFold` **log loss** on the training trips.
  Log loss was chosen over accuracy because it is smooth and ranks probability
  quality, which is what the downstream HMM consumes as emission scores.
- **Trials:** see `results/bayes_opt_trials.csv` (every trial with its objective).
- **Best configuration:** 600 trees, lr=0.0101, max_depth=7, min_child_weight=6.13,
  subsample=0.792, colsample_bytree=0.739, reg_lambda=0.0535, gamma=0.0320.
- **Budget note:** the search uses 3 folds rather than 5; the winning
  configuration is afterwards refit on **all** 1,224,694 reliable training pairs.
- **HMM parameters** (`w_emit=0.5`, `beta=10.0`, `jump_pen=8.0`, `back_pen=2.0`)
  were tuned on **out-of-fold training probabilities** — the student-test
  partition was never used for tuning.

---

## Reproduce

```bash
python src/02_train_compare.py data/features data/processed results models   # ~20 min
python src/03_bayes_opt.py   data/features data/processed results models   # ~20 min
```

Both depend on the feature tables from Part 2 and the decoder in Part 4.

---

## Integrity notes

- Evaluator partition never read.
- Student-test partition never fitted on.
- No metric reported here was entered by hand; each comes from a run of the
  commands above and is stored in `results/`.
- Full-route reconstruction accuracy does not belong to this part — Viterbi
  numbers are produced by the Part 4 decoder and reported for completeness.
