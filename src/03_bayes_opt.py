"""Bayesian hyperparameter optimisation of the selected model (XGBoost) + final model + HMM tuning.
1. Optuna TPE (Bayesian) search. Objective = mean 3-fold GroupKFold log loss on TRAIN trips.
2. Retrain final XGBoost on ALL reliable training pairs with the best parameters.
3. Tune the HMM parameters on OUT-OF-FOLD training probabilities (no student-test data used).
4. Final evaluation on the student-test trips.
Usage: python src/03_bayes_opt.py data/features data/processed results models
"""
import sys, os, json, time
import numpy as np, pandas as pd, joblib, optuna
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (log_loss, f1_score, precision_score, recall_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES
from hmm import viterbi, route_continuity

F, D, R, M = sys.argv[1:5]
SEED = 42
rd = lambda f: pd.read_csv(f"{D}/{f}", dtype={"trip_id": str})
full = pd.read_parquet(f"{F}/model_table_train.parquet")
full = full[full.label_reliable == True].reset_index(drop=True)
full["gid"] = full.trip_id + "_" + full.obs_idx.astype(str)
rng = np.random.default_rng(SEED)
tr = full[full.gid.isin(rng.choice(full.gid.unique(), 30000, replace=False))].reset_index(drop=True)
X, y, g = tr[FEATURES].values, tr.is_true_arc.values, tr.trip_id.values
folds = list(GroupKFold(3).split(X, y, g))


def objective(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
         "max_depth": trial.suggest_int("max_depth", 3, 10),
         "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
         "gamma": trial.suggest_float("gamma", 1e-4, 5, log=True)}
    ll = []
    for a, b in folds:
        m = XGBClassifier(**p, tree_method="hist", eval_metric="logloss", n_jobs=2,
                          random_state=SEED).fit(X[a], y[a])
        ll.append(log_loss(y[b], m.predict_proba(X[b])[:, 1]))
    return float(np.mean(ll))


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=30)
trials = study.trials_dataframe(); trials.to_csv(f"{R}/bayes_opt_trials.csv", index=False)
best = study.best_params
print("best", best, round(study.best_value, 5), flush=True)
plt.figure(figsize=(6, 3.5))
plt.plot(trials.value, "o", alpha=.5, label="trial"); plt.plot(trials.value.cummin(), "r-", label="best so far")
plt.xlabel("trial"); plt.ylabel("3-fold CV log loss"); plt.legend()
plt.title("Bayesian optimisation of XGBoost (Optuna TPE)"); plt.tight_layout()
plt.savefig(f"{R}/plots/bayes_opt_history.png", dpi=110); plt.close()
try:
    imp = optuna.importance.get_param_importances(study)
    pd.Series(imp).sort_values().plot.barh(figsize=(5, 3), title="Hyperparameter importance")
    plt.tight_layout(); plt.savefig(f"{R}/plots/bayes_param_importance.png", dpi=110); plt.close()
except Exception as e:
    print("importance skipped", e)

make = lambda: XGBClassifier(**best, tree_method="hist", eval_metric="logloss", n_jobs=2, random_state=SEED)
XA, yA, gA = full[FEATURES].values, full.is_true_arc.values, full.trip_id.values
t0 = time.time(); final = make().fit(XA, yA); t_train = time.time() - t0
joblib.dump(final, f"{M}/final_model.joblib")
json.dump({"model": "XGBoost", "features": FEATURES, "params": best}, open(f"{M}/final_model_config.json", "w"), indent=1)

# ---- HMM tuning on out-of-fold probabilities of training trips
oof = np.zeros(len(full))
for a, b in GroupKFold(5).split(XA, yA, gA):
    oof[b] = make().fit(XA[a], yA[a]).predict_proba(XA[b])[:, 1]
full["p"] = oof
pts_tr, lab_tr, arcs_tr = rd("points_train_features.csv"), rd("points_train_labels.csv"), rd("arcs_train_features.csv")
cand_all = pd.read_parquet(f"{F}/model_table_train.parquet")
cand_all = cand_all.merge(full[["trip_id", "obs_idx", "arc_id", "p"]], on=["trip_id", "obs_idx", "arc_id"], how="left")
cand_all["p"] = cand_all.p.fillna(0.15)        # pairs of unreliable points: neutral prior
sizes = pts_tr.groupby("trip_id").size()
tune_trips = sorted(sizes[sizes < 4000].index)[:15]


def run(cand, pts, lab, arcs, trips, hp):
    acc, cont = [], []
    for tid in trips:
        c = cand[cand.trip_id == tid]; p_ = pts[pts.trip_id == tid]; a = arcs[arcs.trip_id == tid]
        seq = viterbi(c, p_, a, c.p.values, hp)
        l = lab[lab.trip_id == tid]; rel = l.label_reliable.values.astype(bool)
        acc.append((seq[l.obs_idx.values[rel]] == l.label_arc_id.values[rel]).mean())
        cont.append(route_continuity(seq, a))
    return float(np.mean(acc)), float(np.mean(cont))


grid = []
for w in [0.5, 1.0, 2.0]:
    for beta in [10, 20, 40]:
        for jp in [2.0, 4.0, 8.0]:
            hp = {"w_emit": w, "beta": beta, "jump_pen": jp, "back_pen": 2.0}
            acc, cont = run(cand_all, pts_tr, lab_tr, arcs_tr, tune_trips, hp)
            grid.append({**hp, "train_oof_viterbi_top1": acc, "train_oof_continuity": cont})
            print(hp, round(acc, 4), round(cont, 4), flush=True)
grid = pd.DataFrame(grid); grid.to_csv(f"{R}/hmm_grid_search.csv", index=False)
b = grid.sort_values("train_oof_viterbi_top1").iloc[-1]
hp = {k: float(b[k]) for k in ["w_emit", "beta", "jump_pen", "back_pen"]}
json.dump(hp, open(f"{M}/hmm_params.json", "w"), indent=1)

# ---- final student-test evaluation
te = pd.read_parquet(f"{F}/model_table_student_test.parquet")
pts_te, lab_te, arcs_te = rd("points_student_test_features.csv"), rd("points_student_test_labels.csv"), rd("arcs_student_test_features.csv")
t0 = time.time(); p = final.predict_proba(te[FEATURES].values)[:, 1]; t_inf = time.time() - t0
te["p"] = p; yh = (p >= .5).astype(int)
best_c = te.loc[te.groupby(["trip_id", "obs_idx"]).p.idxmax()]
per = []
for tid in sorted(pts_te.trip_id.unique()):
    a, c = run(te, pts_te, lab_te, arcs_te, [tid], hp)
    per.append({"trip_id": tid, "classifier_top1": best_c[best_c.trip_id == tid].is_true_arc.mean(),
                "viterbi_top1": a, "route_continuity": c})
per = pd.DataFrame(per); per.to_csv(f"{R}/final_student_test_per_trip.csv", index=False)
res = {"model": "XGBoost (Bayes-tuned) + HMM/Viterbi", "best_params": best, "cv_log_loss_best": study.best_value,
       "hmm_params": hp, "test_accuracy": accuracy_score(te.is_true_arc, yh),
       "test_macro_f1": f1_score(te.is_true_arc, yh, average="macro"),
       "test_precision_macro": precision_score(te.is_true_arc, yh, average="macro"),
       "test_recall_macro": recall_score(te.is_true_arc, yh, average="macro"),
       "test_log_loss": log_loss(te.is_true_arc, np.clip(p, 1e-6, 1 - 1e-6)),
       "test_top1_classifier": best_c.is_true_arc.mean(),
       "test_viterbi_top1": per.viterbi_top1.mean(), "test_route_continuity": per.route_continuity.mean(),
       "final_train_rows": len(full), "train_time_s": t_train, "inference_s_all_test_rows": t_inf}
json.dump(res, open(f"{R}/final_results.json", "w"), indent=1, default=float)
ConfusionMatrixDisplay(confusion_matrix(te.is_true_arc, yh), display_labels=["wrong road", "true road"]).plot(
    cmap="Greens", values_format="d")
plt.title("Final tuned XGBoost - student test"); plt.tight_layout()
plt.savefig(f"{R}/plots/cm_final_model.png", dpi=110); plt.close()
pd.Series(final.feature_importances_, FEATURES).sort_values().plot.barh(
    figsize=(6, 4.5), title="Final XGBoost feature importance (gain)")
plt.tight_layout(); plt.savefig(f"{R}/plots/feature_importance.png", dpi=110); plt.close()
x = np.arange(len(per)); plt.figure(figsize=(8, 3.5))
plt.bar(x - .2, per.classifier_top1, .4, label="classifier only"); plt.bar(x + .2, per.viterbi_top1, .4, label="after Viterbi")
plt.xticks(x, per.trip_id.str[-2:]); plt.ylim(.6, 1); plt.xlabel("student-test trip"); plt.ylabel("Top-1 accuracy")
plt.legend(); plt.title("Effect of the HMM per trip"); plt.tight_layout()
plt.savefig(f"{R}/plots/viterbi_gain_per_trip.png", dpi=110); plt.close()
print(json.dumps(res, indent=1, default=float))
