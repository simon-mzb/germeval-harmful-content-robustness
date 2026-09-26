"""
tml_components.py -- the classical machine-learning component of the pipeline (E2).

One shared feature space, three classifiers (linear SVM, XGBoost, LightGBM),
all speaking the component probability interface. Section 3.2 fixes the
representation as TF-IDF over word *and* character n-grams, lowercased, which
is deliberately not inherited from the organiser baseline.

Protocol inside one CV fold
---------------------------
The fold training portion is split, stratified, into three disjoint parts:

    fit (80%)  ->  fits every grid configuration
    select (10%) -> scores them; the best macro-F1 wins; also the early-stopping
                    set for the boosted trees
    calibrate (10%) -> Platt scaling for the SVM, then temperature scaling for
                    every component

The validation part of the fold is touched by nothing except the final
prediction, so hyperparameter selection and calibration are both nested inside
the fold. Pooled out-of-fold scores are therefore not optimistically biased by
either step.

Two recorded deviations from `configs/e2_matrix.yaml` v1.0, both measured
rather than assumed:

* `n_estimators` is not a grid axis. Both boosted models run with a cap of
  1000 rounds and early stopping (patience 30) on the selection slice, which
  decides the number of trees per fit instead of per grid point. This is
  strictly better than choosing between 300 and 800 globally, and it halves
  the grid.
* The grids are consequently 4 points for the tree models and 3 for the SVM.
  The classical arm is not computationally trivial: a
  single four-class XGBoost fit on the shared feature space takes ~80 s on
  this CPU, so the fully nested protocol above with the original 8-point grid
  would have cost several hours per subtask.

Imbalance handling is deliberately absent here: E2 measures the **reference
condition**, and class weighting, focal loss and oversampling are the
E4a comparison against exactly this cell. The consequence is stated where it
matters -- the organiser DBO baseline uses balanced class weights, so the E2
figures are not a like-for-like comparison with it.

Usage
-----
from src.tml_components import TML_COMPONENTS, make_train_fn, predict_proba_fn
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from src.calibration import apply_temperature, fit_temperature
from src.imbalance import balanced_weights, oversample_indices
from src.preprocessing import clean_default


# ---------------------------------------------------------------------------
# Shared feature space (3.2: word AND character n-grams, lowercased)
# ---------------------------------------------------------------------------

FEATURE_SPEC = {
    "word_ngram_range": (1, 2),
    "char_ngram_range": (3, 5),
    "char_analyzer": "char_wb",
    "lowercase": True,          # applied by clean_default, not by the vectoriser
    "min_df": 2,
    "sublinear_tf": True,
}

# Inner split of a fold training portion: fit / select / calibrate.
INNER_SPLIT = {"fit": 0.80, "select": 0.10, "calibrate": 0.10}

EARLY_STOPPING_ROUNDS = 30
MAX_BOOSTING_ROUNDS = 1000


def build_vectoriser() -> FeatureUnion:
    """The shared TF-IDF space; refitted inside every fold on its fit part."""
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=FEATURE_SPEC["word_ngram_range"],
            min_df=FEATURE_SPEC["min_df"],
            sublinear_tf=FEATURE_SPEC["sublinear_tf"],
        )),
        ("char", TfidfVectorizer(
            analyzer=FEATURE_SPEC["char_analyzer"],
            ngram_range=FEATURE_SPEC["char_ngram_range"],
            min_df=FEATURE_SPEC["min_df"],
            sublinear_tf=FEATURE_SPEC["sublinear_tf"],
        )),
    ])


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

GRIDS: dict[str, list[dict[str, Any]]] = {
    # C = 30 added after measurement: with the original {0.1, 1, 10} the winner
    # sat on the upper grid edge (C = 10 won 14/15 folds on C2A and on VIO), so
    # the grid, not the data, was deciding. Probing {..., 30, 100} showed 30 wins
    # 3/5 folds on C2A, 1/5 on DBO and ties C = 10 on VIO, while **100 never wins
    # strictly** -- the maximum is now interior, which is the property that makes
    # the grid defensible. 100 is also excluded on convergence grounds: it needs
    # 4984 of the 5000 allowed liblinear iterations on DBO, i.e. it would be one
    # step from a silently unconverged fit. C = 30 peaks at 1778.
    "tml_svm": [{"C": c} for c in (0.1, 1.0, 10.0, 30.0)],
    "tml_xgboost": [
        {"max_depth": d, "learning_rate": lr}
        for d in (4, 8) for lr in (0.05, 0.1)
    ],
    "tml_lightgbm": [
        {"num_leaves": n, "learning_rate": lr}
        for n in (31, 127) for lr in (0.05, 0.1)
    ],
}

TML_COMPONENTS = tuple(GRIDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _texts(df: pd.DataFrame) -> list[str]:
    return clean_default(df["description"]).tolist()


def _stratifiable(y, n_parts: int) -> bool:
    """Stratification needs at least one member of every class in every part."""
    _, counts = np.unique(y, return_counts=True)
    return bool(counts.min() >= n_parts)


def _three_way_split(df: pd.DataFrame, seed: int):
    """
    Split a fold training portion into fit / select / calibrate, stratified.

    Falls back to an unstratified split only if a class is too thin to appear
    in every part; that fallback is recorded in the fitted component so that it
    cannot silently change what the numbers mean.
    """
    y = df["label"].values
    strat = y if _stratifiable(y, 3) else None
    rest_frac = INNER_SPLIT["select"] + INNER_SPLIT["calibrate"]
    fit_df, rest_df = train_test_split(
        df, test_size=rest_frac, stratify=strat, random_state=seed
    )
    y_rest = rest_df["label"].values
    strat_rest = y_rest if _stratifiable(y_rest, 2) else None
    sel_df, cal_df = train_test_split(
        rest_df,
        test_size=INNER_SPLIT["calibrate"] / rest_frac,
        stratify=strat_rest,
        random_state=seed,
    )
    return (fit_df.reset_index(drop=True),
            sel_df.reset_index(drop=True),
            cal_df.reset_index(drop=True),
            strat is not None and strat_rest is not None)


def _align_columns(proba: np.ndarray, present: Sequence[int], n_classes: int) -> np.ndarray:
    """
    Scatter an estimator probability matrix into the global classes_ ordering.

    `present` holds the global indices the estimator actually saw, in the order
    its own columns use. A class missing from a fit part gets a zero column
    rather than shifting every column to its left.
    """
    if list(present) == list(range(n_classes)):
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for col, gidx in enumerate(present):
        full[:, gidx] = proba[:, col]
    return full


def _fit_estimator(component: str, params: dict, seed: int, n_jobs: int,
                   X_fit, y_fit, X_sel, y_sel, sample_weight=None):
    """Fit one grid point; the boosted models early-stop on the selection slice.

    `sample_weight` carries E4a's class weighting as per-item weights, which all
    three estimators accept natively. None is the E2 path, and passing None is
    exactly what each `fit` defaults to.
    """
    if component == "tml_svm":
        est = LinearSVC(C=params["C"], max_iter=5000, random_state=seed)
        est.fit(X_fit, y_fit, sample_weight=sample_weight)
        return est

    if component == "tml_xgboost":
        import xgboost as xgb

        n_classes_fit = len(np.unique(y_fit))
        est = xgb.XGBClassifier(
            n_estimators=MAX_BOOSTING_ROUNDS,
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            tree_method="hist",
            max_bin=64,
            colsample_bytree=0.3,
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=0,
            eval_metric="mlogloss" if n_classes_fit > 2 else "logloss",
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        )
        est.fit(X_fit, y_fit, sample_weight=sample_weight,
                eval_set=[(X_sel, y_sel)], verbose=False)
        return est

    if component == "tml_lightgbm":
        import lightgbm as lgb

        est = lgb.LGBMClassifier(
            n_estimators=MAX_BOOSTING_ROUNDS,
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            n_jobs=n_jobs,
            random_state=seed,
            verbose=-1,
        )
        est.fit(
            X_fit, y_fit,
            sample_weight=sample_weight,
            eval_set=[(X_sel, y_sel)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                       lgb.log_evaluation(0)],
        )
        return est

    raise ValueError("Unknown TML component: {!r}".format(component))


def _raw_proba(component: str, est, X, present: Sequence[int], n_classes: int) -> np.ndarray:
    """Probabilities before temperature scaling, in the global class ordering."""
    if component == "tml_svm":
        raise RuntimeError("SVM probabilities come from its Platt calibrator.")
    return _align_columns(est.predict_proba(X), present, n_classes)


# ---------------------------------------------------------------------------
# The fitted component
# ---------------------------------------------------------------------------

class FittedTMLComponent:
    """
    A classical component fitted on one fold training portion.

    Carries everything the probability interface needs: the vectoriser, the
    selected estimator, the SVM Platt calibrator where applicable, the fitted
    temperature, and the record of how the configuration was chosen.
    """

    def __init__(self, component: str, classes: list, record: dict[str, Any]):
        self.component = component
        self.classes_ = list(classes)
        self.record = record
        self.vectoriser_: FeatureUnion | None = None
        self.estimator_ = None
        self.platt_ = None
        self.present_: list[int] = []
        self.temperature_: float = 1.0

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """(n_items, n_classes) in the global classes_ ordering, rows summing to 1."""
        X = self.vectoriser_.transform(_texts(df))
        if self.component == "tml_svm":
            scores = self.estimator_.decision_function(X)
            raw = _align_columns(self.platt_.predict_proba(scores),
                                 self.present_, len(self.classes_))
        else:
            raw = _raw_proba(self.component, self.estimator_, X,
                             self.present_, len(self.classes_))
        return apply_temperature(raw, self.temperature_)


def _fit_component(
    component: str,
    train_df: pd.DataFrame,
    classes: list,
    *,
    seed: int,
    n_jobs: int,
    condition: str = "none",
    grid: list[dict[str, Any]] | None = None,
) -> FittedTMLComponent:
    """Fit one component on a fold training portion under the nested protocol.

    `condition` is an E4a imbalance condition (src/imbalance.py) and `grid` a
    one-cell override carrying E4's fixed configuration. Both default to the E2
    path. Oversampling duplicates ROWS OF THE FEATURE MATRIX: the vectoriser is
    fitted on the natural fit slice, so a duplicated text changes how often a
    model sees an item, not the vocabulary or the idf it is described with.
    """
    class_index = {c: i for i, c in enumerate(classes)}
    fit_df, sel_df, cal_df, stratified = _three_way_split(train_df, seed)

    vec = build_vectoriser()
    X_fit = vec.fit_transform(_texts(fit_df))
    X_sel = vec.transform(_texts(sel_df))
    X_cal = vec.transform(_texts(cal_df))

    y_fit = np.array([class_index[v] for v in fit_df["label"].values])
    y_sel = np.array([class_index[v] for v in sel_df["label"].values])
    y_cal = np.array([class_index[v] for v in cal_df["label"].values])

    # Estimators are fitted on the labels actually present in the fit part;
    # `present` maps their column order back onto the global ordering.
    present = sorted(np.unique(y_fit).tolist())
    remap = {g: i for i, g in enumerate(present)}
    y_fit_local = np.array([remap[v] for v in y_fit])
    y_sel_local = np.array([remap.get(v, -1) for v in y_sel])
    keep_sel = y_sel_local >= 0

    n_fit_trained = int(len(y_fit_local))
    sample_weight = None
    if condition == "class_weighting":
        sample_weight = balanced_weights(y_fit_local, len(present))[y_fit_local]
    elif condition == "random_oversampling":
        idx = oversample_indices(y_fit_local, seed)
        X_fit = X_fit[idx]
        y_fit_local = y_fit_local[idx]
        n_fit_trained = int(len(idx))
    elif condition != "none":
        raise ValueError(
            "{} cannot train under {!r}: the protocol defines class weighting and "
            "oversampling for the classical models and nothing else".format(
                component, condition))

    selection: list[dict[str, Any]] = []
    best = None
    for params in (grid if grid is not None else GRIDS[component]):
        est = _fit_estimator(component, params, seed, n_jobs,
                             X_fit, y_fit_local,
                             X_sel[keep_sel], y_sel_local[keep_sel],
                             sample_weight=sample_weight)
        pred_local = est.predict(X_sel[keep_sel])
        score = float(f1_score(y_sel_local[keep_sel], pred_local,
                               average="macro", zero_division=0))
        entry = {"params": dict(params), "selection_macro_f1": score}
        if hasattr(est, "best_iteration") and est.best_iteration is not None:
            entry["best_iteration"] = int(est.best_iteration)
        elif hasattr(est, "best_iteration_") and est.best_iteration_ is not None:
            entry["best_iteration"] = int(est.best_iteration_)
        selection.append(entry)
        if best is None or score > best[0]:
            best = (score, params, est)

    best_score, best_params, best_est = best
    model = FittedTMLComponent(component, classes, {})
    model.vectoriser_ = vec
    model.estimator_ = best_est
    model.present_ = present

    # Platt scaling turns SVM margins into probabilities (3.2, lin2007). It is
    # fitted on the calibration slice, which the selection step did not use.
    if component == "tml_svm":
        y_cal_local = np.array([remap.get(v, -1) for v in y_cal])
        keep_cal = y_cal_local >= 0
        model.platt_ = PlattScaler(len(present)).fit(
            best_est.decision_function(X_cal[keep_cal]), y_cal_local[keep_cal]
        )
        raw_cal = _align_columns(
            model.platt_.predict_proba(best_est.decision_function(X_cal)),
            present, len(classes),
        )
    else:
        raw_cal = _raw_proba(component, best_est, X_cal, present, len(classes))

    # Temperature scaling on top, on the same held-out calibration slice, so
    # that every component in the pipeline carries the same post-hoc treatment
    # (configs/e2_matrix.yaml, calibration.method). For the SVM the Platt step
    # has already used this slice, so its temperature is expected near 1.
    cal_labels = [classes[i] for i in y_cal]
    temp = fit_temperature(raw_cal, cal_labels, classes)
    model.temperature_ = temp["temperature"]

    model.record = {
        "component": component,
        "selected_params": dict(best_params),
        "selection_macro_f1": best_score,
        "selection_grid": selection,
        "temperature": temp,
        "platt_fallback_classes": (
            [str(classes[present[j]]) for j in model.platt_.fallback_]
            if model.platt_ is not None else []
        ),
        "grid_overridden": grid is not None,
        "imbalance_condition": condition,
        "n_fit": int(len(fit_df)),
        "n_fit_trained": n_fit_trained,
        "n_select": int(len(sel_df)),
        "n_calibrate": int(len(cal_df)),
        "n_features": int(X_fit.shape[1]),
        "classes_present_in_fit": [str(classes[i]) for i in present],
        "inner_split_stratified": bool(stratified),
    }
    return model


# ---------------------------------------------------------------------------
# run_cv interface
# ---------------------------------------------------------------------------

def make_train_fn(component: str, classes: list, *, seed: int, n_jobs: int = -1,
                  condition: str = "none",
                  grid: list[dict[str, Any]] | None = None):
    """
    Build a train_fn for harness.run_cv.

    Everything nested inside the fold -- the three-way inner split, the grid,
    Platt scaling and the temperature -- happens here, so run_cv itself stays
    agnostic about what a component is.
    """
    if component not in GRIDS:
        raise ValueError("Unknown TML component: {!r}".format(component))

    def train_fn(train_df: pd.DataFrame) -> FittedTMLComponent:
        return _fit_component(component, train_df, classes, seed=seed, n_jobs=n_jobs,
                              condition=condition, grid=grid)

    return train_fn


def predict_proba_fn(model: FittedTMLComponent, df: pd.DataFrame, classes) -> np.ndarray:
    """predict_proba_fn for harness.run_cv; asserts the declared class ordering."""
    if list(classes) != list(model.classes_):
        raise ValueError(
            "classes_ ordering mismatch: run_cv passed {!r}, component holds {!r}.".format(
                list(classes), list(model.classes_)
            )
        )
    return model.predict_proba(df)


# ---------------------------------------------------------------------------
# Platt scaling for the SVM
# ---------------------------------------------------------------------------

class PlattScaler:
    """
    One-vs-rest Platt scaling on SVM decision values (3.2, lin2007).

    A linear SVM optimises a hinge loss, so its margins are not probabilities;
    3.2 commits to recovering a distribution from them by fitting a sigmoid on
    held-out data. One sigmoid is fitted per class against the one-vs-rest
    target and the resulting scores are normalised to sum to one.

    sklearn's CalibratedClassifierCV is not used here: with a prefit estimator
    it still routes through cross_val_predict, which fails outright when the
    calibration slice happens to miss one of the estimator's classes. At the
    class sizes in this project (DBO subversive contributes roughly five items
    to a calibration slice) that is a live failure mode rather than a corner
    case, so the sigmoid is fitted directly and a class the slice cannot
    support falls back to the untransformed margin, recorded in `fallback_`.
    """

    def __init__(self, n_local_classes: int):
        self.n_local_classes = n_local_classes
        self.coefs_: list[tuple[float, float]] = []
        self.fallback_: list[int] = []

    @staticmethod
    def _as_matrix(scores: np.ndarray, k: int) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim == 1:            # binary LinearSVC: margin for class 1
            return np.column_stack([-scores, scores]) if k == 2 else scores.reshape(-1, 1)
        return scores

    def fit(self, scores: np.ndarray, y_local: np.ndarray) -> "PlattScaler":
        from sklearn.linear_model import LogisticRegression

        S = self._as_matrix(scores, self.n_local_classes)
        for j in range(S.shape[1]):
            target = (y_local == j).astype(int)
            if len(np.unique(target)) < 2:
                self.coefs_.append((1.0, 0.0))
                self.fallback_.append(j)
                continue
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(S[:, [j]], target)
            self.coefs_.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        S = self._as_matrix(scores, self.n_local_classes)
        p = np.empty_like(S, dtype=np.float64)
        for j, (a, b) in enumerate(self.coefs_):
            p[:, j] = 1.0 / (1.0 + np.exp(-(a * S[:, j] + b)))
        p = np.clip(p, 1e-12, None)
        return p / p.sum(axis=1, keepdims=True)
