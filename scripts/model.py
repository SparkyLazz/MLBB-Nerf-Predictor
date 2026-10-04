"""Train and apply a real classifier for "does this hero get nerfed in the next patch".

This sits alongside the original heuristic rather than replacing it. ``scripts/predict.py`` and
its 0.45/0.55 formula are untouched -- that file says not to reweight it, and nothing here does.
What this adds is the other option: learn the weights from the archive instead of guessing them,
across all 61 trend features from scripts/features.py rather than two.

Two candidates are cross-validated and the better one is kept:

  logreg   median-impute -> standardise -> logistic regression (class-balanced). Honest and
           readable with very little data, and its coefficients say what it is keying on.
  gbm      histogram gradient boosting. Handles NaN natively, finds interactions ("high win rate
           AND rising AND heavily banned"), needs more data before it beats logreg.

Validation is leave-one-patch-out: a fold holds out an entire patch window. Anything else leaks,
because 20 daily snapshots of the same hero inside one patch are nearly the same row -- a random
split would score ~0.99 and mean nothing. The metric is average precision, since only a handful
of ~133 heroes get nerfed per patch and ranking the top few is the whole job.

Honest limits, printed by the model report too:
  - fewer than MIN_PATCHES_TO_TRAIN labeled patch windows -> refuses to train and says what is
    missing. That is the current state of this repo until more patches are recorded.
  - fewer than MIN_PATCHES_FOR_CV windows -> trains, but reports scores as indicative only.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as ds  # noqa: E402
import features as featmod  # noqa: E402

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "nerf_model.joblib")

MIN_PATCHES_TO_TRAIN = 2
MIN_PATCHES_FOR_CV = 4
TOP_K = 10  # precision@k -- a nerf list is usually single digits, so the top 10 is the useful window


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "scikit-learn is not installed. Run `pip install -r requirements.txt` "
            "(menu option 15) and try again."
        ) from exc


def build_estimators() -> dict:
    _require_sklearn()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5)),
        ]),
        "gbm": Pipeline([
            ("clf", HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=0,
            )),
        ]),
    }


def usable_features(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into (usable, dropped).

    A column that is entirely NaN or constant across the labeled rows carries no information,
    and the histogram GBM cannot even bin an all-NaN column. The cross-rank spreads are exactly
    this while only one rank bracket is being collected -- they start carrying signal the day a
    second rank enters the archive, and the next retrain picks them up automatically.
    """
    usable, dropped = [], []
    for col in X.columns:
        series = X[col]
        if series.notna().sum() == 0 or series.nunique(dropna=True) <= 1:
            dropped.append(col)
        else:
            usable.append(col)
    return usable, dropped


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    top = np.argsort(-scores)[:k]
    return float(y_true[top].sum() / k)


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> float:
    total = y_true.sum()
    if total == 0:
        return float("nan")
    k = min(k, len(y_true))
    top = np.argsort(-scores)[:k]
    return float(y_true[top].sum() / total)


def _per_snapshot_at_k(y: np.ndarray, scores: np.ndarray, snapshots: np.ndarray, k: int = TOP_K) -> tuple[float, float]:
    """precision@k / recall@k computed one snapshot at a time, then averaged.

    Pooling the whole fold would be wrong: a patch window holds many daily snapshots of the same
    133 heroes, so a pooled "top 10 rows" can be two heroes on five days each. The tool is used
    one day at a time -- rank today's heroes, read the top 10 -- so that is what gets measured.
    """
    precisions, recalls = [], []
    for snap in np.unique(snapshots):
        m = snapshots == snap
        if y[m].sum() == 0:
            continue
        precisions.append(precision_at_k(y[m], scores[m], k))
        recalls.append(recall_at_k(y[m], scores[m], k))
    if not precisions:
        return float("nan"), float("nan")
    return float(np.mean(precisions)), float(np.mean(recalls))


def cross_validate(
    estimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    snapshots: np.ndarray,
) -> dict:
    """Leave-one-patch-out scores. A fold is skipped when either half has only one class."""
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut

    rows = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        est = clone(estimator)
        est.fit(X.iloc[train_idx], y_tr, clf__sample_weight=weights[train_idx])
        p = est.predict_proba(X.iloc[test_idx])[:, 1]
        prec_k, rec_k = _per_snapshot_at_k(y_te, p, snapshots[test_idx])
        rows.append({
            "patch": groups[test_idx][0],
            "n": len(test_idx),
            "positives": int(y_te.sum()),
            "average_precision": average_precision_score(y_te, p),
            "roc_auc": roc_auc_score(y_te, p),
            f"precision_at_{TOP_K}": prec_k,
            f"recall_at_{TOP_K}": rec_k,
        })

    if not rows:
        return {"folds": [], "mean": {}, "note": "no fold had both classes on each side -- too few patches"}

    folds = pd.DataFrame(rows)
    metric_cols = ["average_precision", "roc_auc", f"precision_at_{TOP_K}", f"recall_at_{TOP_K}"]
    return {
        "folds": rows,
        "mean": {c: float(folds[c].mean()) for c in metric_cols},
        "note": "",
    }


def importances(estimator, X: pd.DataFrame, y: np.ndarray, name: str) -> pd.DataFrame:
    """What the fitted model keys on. Coefficients for logreg, permutation for the GBM."""
    if name == "logreg":
        coef = estimator.named_steps["clf"].coef_[0]
        out = pd.DataFrame({"feature": X.columns, "weight": coef})
    else:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(
            estimator, X, y, n_repeats=5, random_state=0, scoring="average_precision",
        )
        out = pd.DataFrame({"feature": X.columns, "weight": r.importances_mean})
    out["abs"] = out["weight"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def train(
    data: pd.DataFrame | None = None,
    model_path: str = MODEL_PATH,
    only: str | None = None,
    save: bool = True,
) -> dict:
    """Fit, cross-validate, pick the better estimator, save a bundle. Returns a report dict."""
    _require_sklearn()
    import joblib
    import sklearn

    if data is None:
        data = ds.build()

    labeled, _ = ds.labeled_split(data)
    patch_windows = sorted(labeled["patch_id"].unique().tolist()) if len(labeled) else []

    if len(patch_windows) < MIN_PATCHES_TO_TRAIN:
        raise RuntimeError(
            f"Only {len(patch_windows)} labeled patch window(s) available "
            f"({', '.join(patch_windows) or 'none'}); need at least {MIN_PATCHES_TO_TRAIN}.\n"
            "A patch window becomes labeled once the NEXT patch is recorded in data/patches.csv\n"
            "with its heroes_nerfed list -- so you need snapshots spanning at least two patches,\n"
            "and the patch after them recorded. Until then use the trend engine\n"
            "(scripts/forecast.py --engine trend), which needs no labels."
        )

    cols, dropped = usable_features(labeled[featmod.feature_columns(data)])
    if not cols:
        raise RuntimeError("No usable feature columns -- every one is empty or constant.")
    X = labeled[cols]
    y = labeled[ds.LABEL].astype(int).to_numpy()
    groups = labeled["patch_id"].to_numpy()
    weights = ds.window_weights(labeled).to_numpy()
    snapshots = labeled["ts"].astype("int64").to_numpy()

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Every labeled row has the same label -- no hero in the labeled windows was nerfed "
            "next patch (or all were). Check heroes_nerfed spellings in data/patches.csv match "
            "the hero names in the snapshots."
        )

    candidates = build_estimators()
    if only:
        if only not in candidates:
            raise RuntimeError(f"--only must be one of {list(candidates)}, got {only!r}")
        candidates = {only: candidates[only]}

    results = {}
    for name, est in candidates.items():
        results[name] = cross_validate(est, X, y, groups, weights, snapshots)

    def rank_key(name: str) -> float:
        mean = results[name]["mean"]
        return mean.get("average_precision", float("-inf")) if mean else float("-inf")

    best_name = max(results, key=rank_key)
    best = candidates[best_name]
    best.fit(X, y, clf__sample_weight=weights)

    imp = importances(best, X, y, best_name)

    bundle = {
        "estimator": best,
        "estimator_name": best_name,
        "features": cols,
        "dropped_features": dropped,
        "label": ds.LABEL,
        "cv": results,
        "importances": imp,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(labeled)),
        "n_positives": int(y.sum()),
        "patch_windows": patch_windows,
        "ranks": sorted(labeled["rank_bracket"].unique().tolist()),
        "cv_reliable": len(patch_windows) >= MIN_PATCHES_FOR_CV,
        "sklearn_version": sklearn.__version__,
    }

    if save:
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        joblib.dump(bundle, model_path)
        bundle["path"] = model_path

    return bundle


def load(model_path: str = MODEL_PATH) -> dict | None:
    """The saved bundle, or None if no model has been trained yet."""
    if not os.path.exists(model_path):
        return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception as exc:
        print(f"WARNING: could not load {model_path} ({exc}); falling back to the trend engine.", file=sys.stderr)
        return None


def predict_proba(bundle: dict, rows: pd.DataFrame) -> np.ndarray:
    """P(nerfed next patch) for feature rows built by scripts/features.py."""
    missing = [c for c in bundle["features"] if c not in rows.columns]
    if missing:
        raise RuntimeError(
            f"Feature rows are missing {len(missing)} column(s) the model was trained on "
            f"(e.g. {missing[:4]}). Rebuild the dataset (menu option 7) or retrain (option 8)."
        )
    return bundle["estimator"].predict_proba(rows[bundle["features"]])[:, 1]


def format_report(bundle: dict, top_features: int = 15) -> str:
    lines = []
    lines.append(f"Model      : {bundle['estimator_name']}")
    lines.append(f"Trained    : {bundle['trained_at']}  (scikit-learn {bundle['sklearn_version']})")
    lines.append(f"Training   : {bundle['n_rows']} labeled rows, {bundle['n_positives']} nerfed")
    lines.append(f"Patches    : {', '.join(bundle['patch_windows'])}")
    lines.append(f"Ranks      : {', '.join(bundle['ranks'])}")
    lines.append(f"Features   : {len(bundle['features'])} used"
                 + (f", {len(bundle['dropped_features'])} dropped as empty/constant"
                    if bundle.get("dropped_features") else ""))
    lines.append("")

    for name, res in bundle["cv"].items():
        marker = " <- selected" if name == bundle["estimator_name"] else ""
        lines.append(f"Leave-one-patch-out CV: {name}{marker}")
        if not res["mean"]:
            lines.append(f"  (no usable folds: {res['note']})")
        else:
            for k, v in res["mean"].items():
                lines.append(f"  {k:<20} {v:.3f}")
        lines.append("")

    if not bundle["cv_reliable"]:
        lines.append(
            f"CAVEAT: only {len(bundle['patch_windows'])} labeled patch window(s). Treat these\n"
            f"        scores as indicative, not measured -- {MIN_PATCHES_FOR_CV}+ windows before\n"
            "        they mean much."
        )
        lines.append("")

    kind = "coefficient" if bundle["estimator_name"] == "logreg" else "permutation importance"
    lines.append(f"Top features by {kind}:")
    for _, row in bundle["importances"].head(top_features).iterrows():
        lines.append(f"  {row['feature']:<24} {row['weight']:+.4f}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the nerf model, or print the saved one's report.")
    ap.add_argument("--report", action="store_true", help="Print the saved model's report and exit.")
    ap.add_argument("--only", choices=["logreg", "gbm"], help="Train just this estimator.")
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.report:
        bundle = load(args.model_path)
        if bundle is None:
            print(f"No model at {args.model_path}. Train one first (menu option 8).", file=sys.stderr)
            sys.exit(1)
        print(format_report(bundle))
        return

    try:
        bundle = train(model_path=args.model_path, only=args.only, save=not args.no_save)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(format_report(bundle))
    if not args.no_save:
        print(f"\nSaved -> {args.model_path}")


if __name__ == "__main__":
    main()
