import os
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from typing import Optional, Dict, Any

MODEL_TAG = "mahi"
OUT_CSV_NAME = f"{MODEL_TAG}.train_metrics_by_cellline_and_tissue.csv"
SAVE_PROBS = True 

# load the pickle file with gene essentiality labels across cell lines
def load_withlabels(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

# list all of the saved cell lines
def infer_num_labels_and_names(obj: dict) -> Tuple[int, List[str]]:
    names = list(map(str, obj.get("_cell_lines", [])))
    if names:
        return len(names), names

# take all the genes, grab their embedding and one label value for the cell line.
# return X and y
def extract_X_y_for_index(obj: dict, label_index: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X_list, y_list, gene_ids = [], [], []
    for gid, entry in obj["data"].items():
        emb = entry.get("input")
        lbl = entry.get("label")
        if emb is None or lbl is None:
            continue
        if len(lbl) <= label_index:
            continue
        emb_arr = np.asarray(emb)
        if emb_arr.ndim != 1:
            raise ValueError(f"Found non-1D embedding with shape {emb_arr.shape}")
        X_list.append(emb_arr)
        y_list.append(int(lbl[label_index]))
        gene_ids.append(str(gid))
    if not X_list:
        return np.empty((0,)), np.empty((0,), dtype=int), []
    # ensure consistent dims
    dim = len(X_list[0])
    for i, v in enumerate(X_list):
        if len(v) != dim:
            raise ValueError(f"Inconsistent embedding dims: first={dim}, idx={i} has {len(v)}")
    return np.stack(X_list, axis=0), np.asarray(y_list, dtype=int), gene_ids

# 5-fold CV with XGBoost 
# handle class imbalance
# return averaged ROCAUC and PRAUC
def cv_eval_xgb(
    X: np.ndarray,
    y: np.ndarray,
    n_splits=5,
    seed=42,
    return_oof: bool = False,
    return_last_model: bool = True,
) -> Tuple[float, float, float, float, Optional[Dict[str, Any]], Optional[XGBClassifier]]:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0 or len(y) < n_splits:
        raise ValueError(f"Insufficient class balance or samples for CV: n={len(y)}, pos={n_pos}, neg={n_neg}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rocs, praucs = [], []
    
    oof_pred = np.full(len(y), np.nan, dtype=float)
    oof_fold = np.full(len(y), -1, dtype=int)
    last_model: Optional[XGBClassifier] = None
    
    base = XGBClassifier(
        booster="gbtree",
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
        random_state=seed,
    )
    for k, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        neg, pos = int((ytr == 0).sum()), int((ytr == 1).sum())
        spw = (neg / pos) if pos > 0 else 1.0

        clf = deepcopy(base)
        clf.set_params(scale_pos_weight=spw)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]

        rocs.append(roc_auc_score(yte, prob))
        praucs.append(average_precision_score(yte, prob))

        oof_pred[te] = prob
        oof_fold[te] = k

        # save the last fold
        if return_last_model and k == (n_splits - 1):
            last_model = clf
        
    summary = {"oof_pred": oof_pred, "oof_fold": oof_fold} if return_oof else None

    return float(np.mean(rocs)), float(np.std(rocs)), float(np.mean(praucs)), float(np.std(praucs)), summary, last_model

def main():
    ap = argparse.ArgumentParser(description="Evaluate Mahi embeddings on gene essentiality (per cell line).")
    ap.add_argument("--out_dir", required=True, help="Base output directory")
    ap.add_argument("--mahi_root", required=True, help="Directory containing Mahi <tissue>.{train,test}.with_labels.pkl")
    ap.add_argument("--mapping_file", required=True, help="Mapping file with columns: cell_line, network")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ap.add_argument("--cell_line", nargs="+", default=None,
                    help="Optional one or more cell lines to run (default: all in mapping)")
    args = ap.parse_args()

    OUT_DIR = Path(args.out_dir)
    TISSUE_ROOT = Path(args.mahi_root)
    MAPPING_FILE = Path(args.mapping_file)

    MAHI_EVAL_DIR = OUT_DIR / "mahi_gene_essentiality_eval"
    MAHI_EVAL_DIR.mkdir(exist_ok=True, parents=True)
    PRED_DIR = MAHI_EVAL_DIR / "cv_preds"
    PRED_DIR.mkdir(exist_ok=True, parents=True)
    TEST_PRED_DIR = MAHI_EVAL_DIR / "test_preds"
    TEST_PRED_DIR.mkdir(exist_ok=True, parents=True)
    OUT_CSV = MAHI_EVAL_DIR / OUT_CSV_NAME
    
    df_map = pd.read_csv(MAPPING_FILE, sep=r"\s+")
    if not {"cell_line", "network"}.issubset(df_map.columns):
        raise ValueError("Mapping file must have columns: cell_line, network")

    if args.cell_line is not None:
        wanted = set(map(str, args.cell_line))
        df_map = df_map[df_map["cell_line"].astype(str).isin(wanted)].copy()
        if df_map.empty:
            raise SystemExit(f"No rows left after filtering for cell_line in {wanted}")

    rows: List[dict] = []
    for _, row in df_map.iterrows():
        cell_line = str(row["cell_line"]).strip()
        tissue = str(row["network"]).strip()
    
        # point to the right pickle file for this tissue
        train_p = TISSUE_ROOT / f"{tissue}.train.with_labels.pkl"
        if not train_p.exists():
            print(f"[miss] {train_p} not found", flush=True)
            continue

        try:
            obj = load_withlabels(train_p)
            n_labels, cell_line_names = infer_num_labels_and_names(obj)
            if n_labels == 0:
                print(f"[warn] {train_p.name}: no _cell_lines found", flush=True)
                continue
    
            # find this cell_line’s index
            try:
                li = cell_line_names.index(cell_line)
            except ValueError:
                print(f"[warn] {train_p.name}: cell_line {cell_line} not found", flush=True)
                continue
    
            X, y, gene_ids = extract_X_y_for_index(obj, li)
            if len(y) == 0:
                print(f"[warn] {cell_line}: no labeled genes; skipping.", flush=True)
                continue

            roc_mean, roc_std, pr_mean, pr_std, oof, last_model = cv_eval_xgb(
                X, y, n_splits=5, seed=args.seed, return_oof=SAVE_PROBS, return_last_model=True,
            )
    
            row = {
                "tissue": tissue,
                "file": train_p.name,
                "cell_line": cell_line,
                "label_index": li,
                "n": int(len(y)),
                "pos": int((y == 1).sum()),
                "neg": int((y == 0).sum()),
                "roc_mean": roc_mean,
                "roc_std": roc_std,
                "pr_mean": pr_mean,
                "pr_std": pr_std,
            }
            rows.append(row)
            print(f"[OK] {tissue}/{cell_line} | n={row['n']} (pos={row['pos']}, neg={row['neg']}) | ROC={roc_mean:.4f} PR={pr_mean:.4f}", flush=True)

            if SAVE_PROBS and oof is not None:
                df_pred = pd.DataFrame({
                    "gene_id": gene_ids,
                    "label": y.astype(int),
                    "oof_pred": oof["oof_pred"],
                    "cv_fold": oof["oof_fold"].astype(int),
                    "cell_line": cell_line,
                    "tissue": tissue,
                    "model": MODEL_TAG,
                })
                pred_path = PRED_DIR / f"{tissue}_{cell_line}_cv_preds.csv"
                df_pred.to_csv(pred_path, index=False)
                print(f"[save] per-gene predictions -> {pred_path}", flush=True)

            if last_model is None:
                print(f"[warn] {tissue}/{cell_line}: last_model is None; skipping test eval.", flush=True)
            else:
                test_p = TISSUE_ROOT / f"{tissue}.test.with_labels.pkl"
                if not test_p.exists():
                    print(f"[miss] TEST {test_p} not found", flush=True)
                else:
                    test_obj = load_withlabels(test_p)
                    n_labels_test, cell_line_names_test = infer_num_labels_and_names(test_obj)
                    if n_labels_test == 0:
                        print(f"[warn] TEST {test_p.name}: no _cell_lines found", flush=True)
                    else:
                        try:
                            li_test = cell_line_names_test.index(cell_line)
                        except ValueError:
                            print(f"[warn] TEST {test_p.name}: cell_line {cell_line} not found", flush=True)
                        else:
                            X_test, y_test, gene_ids_test = extract_X_y_for_index(test_obj, li_test)
                            if len(y_test) == 0:
                                print(f"[warn] TEST {cell_line}: no labeled genes; skipping test eval.", flush=True)
                            else:
                                # the model will evaluate on the test set only
                                test_prob = last_model.predict_proba(X_test)[:, 1]

                                # just for logging
                                test_roc = roc_auc_score(y_test, test_prob)
                                test_pr = average_precision_score(y_test, test_prob)
                                print(
                                    f"[TEST] {tissue}/{cell_line} | n={len(y_test)} | "
                                    f"ROC={test_roc:.4f} PR={test_pr:.4f}",
                                    flush=True,
                                )

                                # save the test predictions
                                # predicted probabilities + labels + gene_id
                                df_test_pred = pd.DataFrame({
                                    "gene_id": gene_ids_test,
                                    "label": y_test.astype(int),
                                    "pred_prob": test_prob,
                                    "cell_line": cell_line,
                                    "tissue": tissue,
                                    "model": MODEL_TAG,
                                })
                                test_pred_path = TEST_PRED_DIR / f"{tissue}_{cell_line}_preds.csv"
                                df_test_pred.to_csv(test_pred_path, index=False)
                                print(f"[save] TEST preds -> {test_pred_path}", flush=True)
    
        except Exception as e:
            print(f"[ERR] {tissue}/{cell_line}: {e}", flush=True)

    if rows:
        df = pd.DataFrame(rows).sort_values(["tissue", "cell_line"])
        df.to_csv(OUT_CSV, index=False, float_format="%.6f")
        print(f"[save] {OUT_CSV}", flush=True)
    else:
        print(f"[warn] {MODEL_TAG}: no valid results.", flush=True)

if __name__ == "__main__":
    main()