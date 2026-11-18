"""Part 3 - Train and compare the five models:
   Logistic Regression | KNN | SVM | Random Forest | XGBoost

Each row = one (GPS point, candidate road) pair. Target = is_true_arc (1 = the correct road).
* 5-fold GroupKFold cross-validation on TRAIN trips (a trip is never split across folds).
* Comparison on the 10 STUDENT-TEST trips.
* Every model's probabilities are also decoded by the HMM/Viterbi (default settings).
Usage: python src/02_train_compare.py data/features data/processed results models
"""
import sys, os, time, json
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, log_loss, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES
from hmm import viterbi, route_continuity

F, D, R, M = sys.argv[1:5]
os.makedirs(f"{R}/plots", exist_ok=True); os.makedirs(M, exist_ok=True)
SEED = 42
N_OBS = 30000      # GPS observations sampled for training (same sample for every model)
N_OBS_SVM = 3000   # RBF-SVM cost grows ~quadratically, so it gets a smaller subset of the same sample

tr = pd.read_parquet(f"{F}/model_table_train.parquet")
tr = tr[tr.label_reliable == True].reset_index(drop=True)
tr["gid"] = tr.trip_id + "_" + tr.obs_idx.astype(str)
rng = np.random.default_rng(SEED)
obs = rng.choice(tr.gid.unique(), N_OBS, replace=False)
tr = tr[tr.gid.isin(obs)].reset_index(drop=True)
svm_obs = set(obs[:N_OBS_SVM])
te = pd.read_parquet(f"{F}/model_table_student_test.parquet")
pts_te = pd.read_csv(f"{D}/points_student_test_features.csv", dtype={"trip_id": str})
lab_te = pd.read_csv(f"{D}/points_student_test_labels.csv", dtype={"trip_id": str})
arcs_te = pd.read_csv(f"{D}/arcs_student_test_features.csv", dtype={"trip_id": str})
print("train rows", len(tr), "| test rows", len(te))

MODELS = {
    "LogisticRegression": ("Linear", "C=1.0, L2 penalty, standardised features",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))),
    "KNN": ("Instance-based", "k=25, distance-weighted, Euclidean, standardised",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(25, weights="distance", n_jobs=2))),
    "SVM": ("Kernel", "RBF kernel, C=1, gamma='scale', Platt probabilities, 3k-obs subset",
        lambda: make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED))),
    "RandomForest": ("Ensemble (bagging)", "200 trees, min_samples_leaf=5, sqrt features",
        lambda: RandomForestClassifier(200, min_samples_leaf=5, n_jobs=2, random_state=SEED)),
    "XGBoost": ("Boosting", "300 trees, lr=0.1, max_depth=6, hist",
        lambda: XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                              eval_metric="logloss", n_jobs=2, random_state=SEED)),
}


def top1(df, p):
    """Observation-level Top-1: the highest-scored candidate is the reference arc."""
    d = df[["trip_id", "obs_idx", "is_true_arc"]].assign(p=p)
    return d.loc[d.groupby(["trip_id", "obs_idx"]).p.idxmax()].is_true_arc.mean()


def metrics(df, p):
    y, yh = df.is_true_arc.values, (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, yh), "macro_f1": f1_score(y, yh, average="macro"),
            "precision_macro": precision_score(y, yh, average="macro", zero_division=0),
            "recall_macro": recall_score(y, yh, average="macro"),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "top1": top1(df, p)}


def decode(p_all, hp=None):
    """HMM/Viterbi on each student-test trip; accuracy over reliable labelled points."""
    acc, cont = [], []
    t = te.assign(p=p_all)
    for tid, pts in pts_te.groupby("trip_id"):
        c = t[t.trip_id == tid]; a = arcs_te[arcs_te.trip_id == tid]
        seq = viterbi(c, pts, a, c.p.values, **({"hp": hp} if hp else {}))
        lab = lab_te[lab_te.trip_id == tid]; rel = lab.label_reliable.values.astype(bool)
        acc.append((seq[lab.obs_idx.values[rel]] == lab.label_arc_id.values[rel]).mean())
        cont.append(route_continuity(seq, a))
    return float(np.mean(acc)), float(np.mean(cont))


rows, log = [], {}
for name, (fam, cfg, make) in MODELS.items():
    print("==", name, flush=True)
    d = tr[tr.gid.isin(svm_obs)].reset_index(drop=True) if name == "SVM" else tr
    X, y, g = d[FEATURES].values, d.is_true_arc.values, d.trip_id.values
    cv = []
    for a_, b_ in GroupKFold(5).split(X, y, g):
        m = make().fit(X[a_], y[a_])
        cv.append(metrics(d.iloc[b_], m.predict_proba(X[b_])[:, 1]))
    cvd = pd.DataFrame(cv)
    t0 = time.time(); m = make().fit(X, y); t_train = time.time() - t0
    trm = metrics(d, m.predict_proba(X)[:, 1])
    t0 = time.time(); p = m.predict_proba(te[FEATURES].values)[:, 1]
    t_inf = (time.time() - t0) / len(te) * 1e6          # ms per 1000 rows
    st = metrics(te, p)
    vit, cont = decode(p)
    joblib.dump(m, f"{M}/{name}.joblib")
    np.save(f"{M}/{name}_student_test_proba.npy", p)
    r = {"model": name, "family": fam, "config": cfg, "train_rows": len(d),
         "cv_macro_f1": cvd.macro_f1.mean(), "cv_macro_f1_std": cvd.macro_f1.std(),
         "cv_top1": cvd.top1.mean(), "cv_log_loss": cvd.log_loss.mean(),
         "train_macro_f1": trm["macro_f1"], "train_top1": trm["top1"],
         **{f"test_{k}": v for k, v in st.items()},
         "viterbi_top1": vit, "route_continuity": cont,
         "train_time_s": t_train, "inference_ms_per_1k_rows": t_inf}
    rows.append(r); log[name] = {"cv_folds": cv, **r}
    print({k: (round(v, 4) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}, flush=True)
    ConfusionMatrixDisplay(confusion_matrix(te.is_true_arc, (p >= .5).astype(int)),
                           display_labels=["wrong road", "true road"]).plot(cmap="Blues", values_format="d")
    plt.title(f"{name} - student test"); plt.tight_layout()
    plt.savefig(f"{R}/plots/cm_{name}.png", dpi=110); plt.close()

res = pd.DataFrame(rows)
res.to_csv(f"{R}/model_comparison.csv", index=False)
json.dump(log, open(f"{R}/experiment_log.json", "w"), indent=1, default=float)

fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
for i, (col, t) in enumerate([("cv_macro_f1", "CV macro F1 (train)"), ("test_macro_f1", "Student-test macro F1"),
                              ("test_top1", "Top-1 (classifier only)"), ("viterbi_top1", "Top-1 after Viterbi")]):
    ax[i].barh(res.model, res[col], color="#4C78A8"); ax[i].set_title(t); ax[i].set_xlim(0.5, 1)
    for j, v in enumerate(res[col]):
        ax[i].text(v + .005, j, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{R}/plots/model_comparison.png", dpi=110); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.scatter(res.inference_ms_per_1k_rows, res.test_macro_f1, s=60)
for r in res.itertuples():
    ax.annotate(r.model, (r.inference_ms_per_1k_rows, r.test_macro_f1), fontsize=8, xytext=(4, 4),
                textcoords="offset points")
ax.set_xscale("log"); ax.set_xlabel("inference time, ms per 1k rows (log)"); ax.set_ylabel("test macro F1")
ax.set_title("Accuracy vs speed"); plt.tight_layout(); plt.savefig(f"{R}/plots/speed_vs_f1.png", dpi=110)
print(res[["model", "cv_macro_f1", "cv_log_loss", "test_macro_f1", "test_log_loss", "test_top1",
           "viterbi_top1", "route_continuity", "train_time_s", "inference_ms_per_1k_rows"]].round(4).to_string())
