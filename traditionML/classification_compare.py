import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    StratifiedKFold,
    GroupKFold,
    cross_validate,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def build_models():
    return {
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True)),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }


def load_feature_matrix(features_csv):
    features_csv = Path(features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {features_csv}")

    def is_numeric_row(row):
        if not row:
            return False
        for cell in row:
            value = cell.strip()
            if value == "":
                return False
            try:
                float(value)
            except ValueError:
                return False
        return True

    with open(features_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(c.strip() for c in row)]

    if not rows:
        raise ValueError(f"Feature CSV is empty: {features_csv}")

    start_idx = 0 if is_numeric_row(rows[0]) else 1
    data_rows = rows[start_idx:]
    if not data_rows:
        raise ValueError(f"No feature rows found in: {features_csv}")

    try:
        X = np.array([[float(c) for c in row] for row in data_rows], dtype=float)
    except ValueError as e:
        raise ValueError(
            f"Non-numeric cell in feature rows of {features_csv}. "
            "Use numeric-only rows (optional one header row)."
        ) from e

    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def append_missing_label_rows(
    label_path: Path,
    n_append: int,
    *,
    pad_fill: str,
    y_existing: np.ndarray,
) -> None:
    """
    Append n_append rows to a label file so its line count can match a longer feature CSV.

    - .txt: one label per line (same string each line when pad_fill is 'last': last label in y_existing).
    - .csv: preserves header if present; new rows duplicate the last data row (pad_fill=='last')
      or use pad_fill as the first column and empty strings for remaining columns.
    """
    label_path = Path(label_path)
    n_append = int(n_append)
    if n_append <= 0:
        return

    if pad_fill == "last":
        if len(y_existing) == 0:
            raise ValueError("Cannot pad labels: no existing labels to take 'last' from.")
        fill_first = str(y_existing[-1])
    else:
        fill_first = pad_fill.strip()

    suffix = label_path.suffix.lower()
    if suffix == ".txt":
        with open(label_path, "a", encoding="utf-8") as fp:
            for _ in range(n_append):
                fp.write(fill_first + "\n")
        return

    if suffix == ".csv":
        with open(label_path, "r", encoding="utf-8", newline="") as fp:
            rows = list(csv.reader(fp))
        if not rows:
            raise ValueError(f"Label CSV is empty, cannot append: {label_path}")

        header_like = rows[0] and rows[0][0].strip().lower() in {"label", "labels", "type", "class"}
        if header_like:
            header, data = rows[0], rows[1:]
        else:
            header, data = None, rows[:]

        if pad_fill == "last":
            template = list(data[-1]) if data else [fill_first]
        else:
            ncol = len(data[-1]) if data else 1
            template = [fill_first] + ([""] * (ncol - 1)) if ncol > 1 else [fill_first]

        tail = [list(template) for _ in range(n_append)]
        out_rows = ([header] if header is not None else []) + data + tail
        tmp = label_path.with_name(label_path.name + ".align_tmp")
        try:
            with open(tmp, "w", encoding="utf-8", newline="") as fp:
                csv.writer(fp).writerows(out_rows)
            tmp.replace(label_path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        return

    raise ValueError(
        f"--align_mismatch pad_file_last only supports .txt / .csv labels; got {label_path}"
    )


def align_feature_label_counts(
    X_part: np.ndarray,
    y_part: np.ndarray,
    label_path: Path,
    *,
    align_mismatch: str,
    pad_label_fill: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    align_mismatch:
      strict — require equal lengths (default).
      pad_memory_last — if more features than labels, pad y by repeating last label (or pad_label_fill);
        if more labels than features, truncate y to len(X) (warning only).
      pad_file_last — if more features than labels, append rows to label_path then reload labels;
        if more labels than features, truncate y in memory (label file unchanged).
    """
    label_path = Path(label_path)
    nx, ny = len(X_part), len(y_part)
    if nx == ny:
        return X_part, y_part

    if align_mismatch == "strict":
        raise ValueError(
            f"Shape mismatch for features vs {label_path.name}: "
            f"{nx} feature rows vs {ny} labels. "
            "Use --align_mismatch pad_memory_last or pad_file_last to reconcile."
        )

    if align_mismatch not in {"pad_memory_last", "pad_file_last"}:
        raise ValueError(f"Unknown align_mismatch: {align_mismatch!r}")

    if nx > ny:
        n_pad = nx - ny
        if ny == 0:
            if pad_label_fill == "last":
                raise ValueError(
                    f"{label_path.name}: 0 labels but {nx} feature rows; "
                    "cannot use --pad_label_fill last. Set --pad_label_fill to an explicit class name."
                )
            fill = pad_label_fill
            y_pad = np.array([fill] * nx, dtype=object)
            print(f"  [align] {label_path.name}: all {nx} labels set to {fill!r} (had no labels).")
            if align_mismatch == "pad_file_last":
                append_missing_label_rows(
                    label_path, nx, pad_fill=pad_label_fill, y_existing=y_pad[:1]
                )
                y_new = load_labels(label_path)
                return X_part, y_new
            return X_part, y_pad

        if align_mismatch == "pad_memory_last":
            fill = str(y_part[-1]) if pad_label_fill == "last" else pad_label_fill
            y_pad = np.concatenate([y_part, np.array([fill] * n_pad, dtype=object)])
            print(
                f"  [align] {label_path.name}: padded {n_pad} label(s) in memory "
                f"(fill={fill!r}) to match {nx} feature rows."
            )
            return X_part, y_pad

        # pad_file_last
        append_missing_label_rows(
            label_path, n_pad, pad_fill=pad_label_fill, y_existing=y_part
        )
        y_new = load_labels(label_path)
        print(
            f"  [align] {label_path.name}: appended {n_pad} row(s) on disk "
            f"(fill={'last row' if pad_label_fill == 'last' else pad_label_fill!r}) "
            f"to match {nx} feature rows."
        )
        if len(y_new) != nx:
            raise RuntimeError(
                f"After appending labels, expected {nx} labels, got {len(y_new)} for {label_path}"
            )
        return X_part, y_new

    # nx < ny: truncate labels to match features (do not shrink label file)
    print(
        f"  [align] {label_path.name}: truncated labels from {ny} to {nx} in memory "
        f"to match feature rows (label file not modified)."
    )
    return X_part, y_part[:nx].copy()


def load_labels(labels_file):
    labels_path = Path(labels_file)
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")

    suffix = labels_path.suffix.lower()

    if suffix == ".csv":
        with open(labels_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = [row for row in reader if row]
        if not rows:
            raise ValueError(f"Label CSV is empty: {labels_path}")

        header_like = rows[0][0].strip().lower() in {"label", "labels", "type", "class"}
        data_rows = rows[1:] if header_like else rows
        labels = [row[0].strip() for row in data_rows if row and row[0].strip()]
    else:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

    if not labels:
        raise ValueError(f"No valid labels loaded from: {labels_path}")
    return np.array(labels)


def label_basename_from_feature_csv_stem(csv_stem: str) -> str:
    """
    MFCC CSV stem like 'id_MFCC' -> label file basename 'id' (strip trailing _MFCC).
    """
    return csv_stem[: -len("_MFCC")] if csv_stem.endswith("_MFCC") else csv_stem


def _resolve_label_path(labels_dir: Path, csv_stem: str) -> Path:
    """
    Prefer labels named after csv stem with _MFCC removed, e.g.
    xxx_MFCC.csv -> xxx.txt. Fallback: full csv stem for older layouts.
    """
    base = label_basename_from_feature_csv_stem(csv_stem)

    candidates = [
        labels_dir / f"{base}.txt",
        labels_dir / f"{base}.csv",
        labels_dir / f"{csv_stem}.txt",
        labels_dir / f"{csv_stem}.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No label file for features stem {csv_stem!r} in {labels_dir}: "
        f"tried {base}.txt / {base}.csv / {csv_stem}.txt / {csv_stem}.csv"
    )


def load_dataset_from_directories(
    features_dir,
    labels_dir,
    *,
    align_mismatch: str = "strict",
    pad_label_fill: str = "last",
):
    """
    Pair features_dir/*.csv with labels_dir/*.txt|.csv:

    Primary: MFCC CSV stem without trailing _MFCC matches label basename
    (e.g. id_MFCC.csv ↔ id.txt). Fallback: same stem as the CSV filename.

    Samples from one feature file share one group id (not split across train/test).

    When row counts differ, see align_mismatch / pad_label_fill (also exposed as CLI flags).
    """
    features_dir = Path(features_dir)
    labels_dir = Path(labels_dir)
    if not features_dir.is_dir():
        raise NotADirectoryError(features_dir)
    if not labels_dir.is_dir():
        raise NotADirectoryError(labels_dir)

    csv_paths = sorted(features_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"No *.csv files in {features_dir}")

    X_chunks = []
    y_list = []
    group_labels = []

    for group_id, csv_path in enumerate(csv_paths):
        stem = csv_path.stem
        label_path = _resolve_label_path(labels_dir, stem)

        X_part = load_feature_matrix(csv_path)
        y_part = load_labels(label_path)
        X_part, y_part = align_feature_label_counts(
            X_part,
            y_part,
            label_path,
            align_mismatch=align_mismatch,
            pad_label_fill=pad_label_fill,
        )

        X_chunks.append(X_part)
        y_list.extend(list(y_part))
        group_labels.extend([group_id] * len(X_part))

    X = np.vstack(X_chunks)
    y = np.array(y_list)
    groups = np.array(group_labels, dtype=int)
    stems = [p.stem for p in csv_paths]
    return X, y, groups, stems


def run_comparison(X, y, test_size=0.2, random_state=42, groups=None):
    if len(X) != len(y):
        raise ValueError(f"Sample count mismatch: X has {len(X)} rows, y has {len(y)} labels.")

    if len(np.unique(y)) < 2:
        raise ValueError("At least 2 classes are required for classification.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if groups is not None:
        if len(groups) != len(X):
            raise ValueError("groups length must match number of samples.")
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError(
                "Need at least 2 CSV files (groups) for group-wise split. "
                "Use single-file mode (--features_csv) or add more paired CSVs."
            )
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y_encoded, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    else:
        _, counts = np.unique(y_encoded, return_counts=True)
        strat = y_encoded if np.min(counts) >= 2 else None
        if strat is None:
            print("Warning: some class has fewer than 2 samples; stratify disabled for split.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=strat,
        )

    models = build_models()
    results = {}

    for name, model in models.items():
        print(f"\n=== Training {name} ===")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Accuracy: {acc:.4f}")
        print(f"Macro-F1: {f1:.4f}")
        print("Detailed Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        results[name] = {
            "accuracy": acc,
            "f1": f1,
            "y_pred": y_pred,
        }

    best_model = max(results, key=lambda k: results[k]["f1"])
    print("\nBest model:", best_model)
    print("Score:", results[best_model])
    print("Best model confusion matrix:")
    print(confusion_matrix(y_test, results[best_model]["y_pred"]))


def run_cross_validation(X, y, cv_folds=5, random_state=42, groups=None):
    """
    Compare models with K-fold CV. Uses GroupKFold when groups is set (each MFCC file = one group),
    else StratifiedKFold on samples.

    Returns:
        best_model_name: str
        cv_macro_f1_means: dict model_name -> mean test macro-F1 across folds
        label_encoder: fitted LabelEncoder on y
    """
    if len(X) != len(y):
        raise ValueError(f"Sample count mismatch: X has {len(X)} rows, y has {len(y)} labels.")
    if len(np.unique(y)) < 2:
        raise ValueError("At least 2 classes are required for classification.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2.")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if groups is not None:
        if len(groups) != len(X):
            raise ValueError("groups length must match number of samples.")
        n_groups = len(np.unique(groups))
        if n_groups < cv_folds:
            raise ValueError(
                f"cv_folds={cv_folds} requires at least that many groups; got {n_groups}."
            )
        cv = GroupKFold(n_splits=cv_folds)
        split_groups = groups
    else:
        _, counts = np.unique(y_encoded, return_counts=True)
        min_class = int(np.min(counts))
        if min_class < cv_folds:
            raise ValueError(
                f"Smallest class count is {min_class}; need >= cv_folds ({cv_folds}) "
                "for StratifiedKFold. Reduce --cv_folds or use more data."
            )
        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=random_state
        )
        split_groups = None

    models = build_models()
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}

    print(f"\n=== {cv_folds}-fold cross-validation ===")
    print(
        "Splitter:",
        "GroupKFold (by feature file)" if groups is not None else "StratifiedKFold",
    )

    summary = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y_encoded,
            cv=cv,
            scoring=scoring,
            groups=split_groups,
            n_jobs=-1,
        )
        acc_mean = float(np.mean(scores["test_accuracy"]))
        acc_std = float(np.std(scores["test_accuracy"]))
        f1_mean = float(np.mean(scores["test_f1_macro"]))
        f1_std = float(np.std(scores["test_f1_macro"]))
        summary.append((name, acc_mean, acc_std, f1_mean, f1_std))
        print(f"\n--- {name} ---")
        print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  Macro-F1: {f1_mean:.4f} ± {f1_std:.4f}")

    print("\n=== CV summary (mean ± std) ===")
    print(f"{'Model':<22} {'Accuracy':>22} {'Macro-F1':>22}")
    for name, acc_mean, acc_std, f1_mean, f1_std in summary:
        print(
            f"{name:<22} {acc_mean:.4f} ± {acc_std:.4f}"
            f"{'':>6}{f1_mean:.4f} ± {f1_std:.4f}"
        )

    best = max(summary, key=lambda row: row[3])
    print(f"\nBest by mean macro-F1: {best[0]}")
    cv_macro_f1_means = {row[0]: row[3] for row in summary}
    return best[0], cv_macro_f1_means, le


def fit_and_save_best_model_bundle(
    X: np.ndarray,
    y_encoded: np.ndarray,
    *,
    best_model_name: str,
    label_encoder: LabelEncoder,
    cv_folds: int,
    cv_macro_f1_means: dict[str, float],
    model_out: Path,
) -> None:
    """Fit the chosen estimator on all data; save joblib bundle for model_predict.py."""
    model = build_models()[best_model_name]
    model.fit(X, y_encoded)
    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "best_model_name": best_model_name,
        "n_features": int(X.shape[1]),
        "cv_folds": cv_folds,
        "cv_macro_f1_means": cv_macro_f1_means,
    }
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)
    print(f"\nSaved best model bundle -> {model_out.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare SVM, RandomForest, LogisticRegression. "
            "Single CSV + labels, or feature/label dirs: id_MFCC.csv pairs with "
            "id.txt in labels_dir. GroupShuffleSplit keeps each CSV wholly in "
            "train or test. Use --cv_folds K for K-fold CV (GroupKFold if grouped). "
            "With --cv_folds and --save_model, fit the CV-best model on all data and "
            "save a joblib bundle for model_predict.py."
        )
    )

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--features_csv",
        default=None,
        help="Single feature matrix CSV (requires --labels_file).",
    )
    grp.add_argument(
        "--features_dir",
        default=None,
        help="Folder of feature *.csv; paired labels drop trailing _MFCC from stem.",
    )

    parser.add_argument(
        "--labels_file",
        default=None,
        help="Labels for --features_csv (.txt one per line, or .csv first column).",
    )
    parser.add_argument(
        "--labels_dir",
        default=None,
        help=(
            "Label folder when using --features_dir. "
            "Pairs id_MFCC.csv with id.txt (or id.csv); falls back to id_MFCC.txt."
        ),
    )
    parser.add_argument(
        "--align_mismatch",
        choices=("strict", "pad_memory_last", "pad_file_last"),
        default="strict",
        help=(
            "When a feature CSV has a different number of rows than its label file: "
            "strict (error); pad_memory_last (repeat last label or --pad_label_fill in RAM); "
            "pad_file_last (append that many rows to the label .txt/.csv on disk, then reload). "
            "If there are more labels than feature rows, labels are truncated in memory only."
        ),
    )
    parser.add_argument(
        "--pad_label_fill",
        type=str,
        default="last",
        metavar="STR",
        help=(
            "With pad_* modes when padding is needed: 'last' repeats the last label row, "
            "else use this exact class string (CSV: first column; extra columns empty)."
        ),
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split proportion.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=None,
        metavar="K",
        help=(
            "If set to K>=2, run K-fold cross-validation for all models instead of "
            "a single holdout split. With --features_dir, uses GroupKFold by file; "
            "with --features_csv, uses StratifiedKFold."
        ),
    )
    parser.add_argument(
        "--save_model",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Requires --cv_folds: after CV, fit the best-by-macro-F1 model on ALL samples "
            "and save joblib to PATH (for traditionML/model_predict.py predict)."
        ),
    )
    args = parser.parse_args()

    if args.save_model is not None and args.cv_folds is None:
        parser.error("--save_model requires --cv_folds (CV is used to pick the best model).")

    def run_eval(X_arr, y_arr, groups=None):
        if args.cv_folds is not None:
            best_name, cv_f1, le = run_cross_validation(
                X_arr,
                y_arr,
                cv_folds=args.cv_folds,
                random_state=args.random_state,
                groups=groups,
            )
            y_enc = le.transform(y_arr)
            if args.save_model is not None:
                fit_and_save_best_model_bundle(
                    X_arr,
                    y_enc,
                    best_model_name=best_name,
                    label_encoder=le,
                    cv_folds=args.cv_folds,
                    cv_macro_f1_means=cv_f1,
                    model_out=args.save_model,
                )
        else:
            run_comparison(
                X_arr,
                y_arr,
                test_size=args.test_size,
                random_state=args.random_state,
                groups=groups,
            )

    if args.features_csv is not None:
        if not args.labels_file:
            parser.error("--labels_file is required with --features_csv")
        X = load_feature_matrix(args.features_csv)
        y = load_labels(args.labels_file)
        X, y = align_feature_label_counts(
            X,
            y,
            Path(args.labels_file),
            align_mismatch=args.align_mismatch,
            pad_label_fill=args.pad_label_fill,
        )
        run_eval(X, y)
    else:
        if not args.labels_dir:
            parser.error("--labels_dir is required with --features_dir")
        X, y, groups, stems = load_dataset_from_directories(
            args.features_dir,
            args.labels_dir,
            align_mismatch=args.align_mismatch,
            pad_label_fill=args.pad_label_fill,
        )
        print(f"Loaded {len(stems)} groups: {', '.join(stems)}")
        print(f"Total samples: {len(X)}\n")
        run_eval(X, y, groups=groups)


if __name__ == "__main__":
    main()
