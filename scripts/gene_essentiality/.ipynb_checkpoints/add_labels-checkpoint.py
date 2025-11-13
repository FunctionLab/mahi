import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import argparse

EXPECTED_DIM: Optional[int] = None

def normalize_entrez_key(k) -> str:
    try:
        return str(int(k))
    except Exception:
        return str(k).strip()

def split_gene_col(col: str) -> Tuple[Optional[str], Optional[str]]:
    # columns like "A1BG_1", "A1CF_29974"
    if col == "cell_line":
        return None, None
    gene, entrez = col.rsplit("_", 1)
    return gene, entrez

def build_entrez_to_col(df: pd.DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for col in df.columns:
        if col == "cell_line":
            continue
        _, entrez = split_gene_col(col)
        if entrez and entrez not in mapping:
            mapping[entrez] = col
    return mapping

def load_any_embedding_dict(pkl_path: Path) -> Dict[str, np.ndarray]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    def _to_vec(x):
        arr = np.asarray(x)
        if arr.ndim != 1:
            raise ValueError(f"Embedding is not 1D (shape={arr.shape}) in {pkl_path.name}")
        return arr

    if isinstance(obj, dict) and "data" not in obj:
        return {normalize_entrez_key(k): _to_vec(v) for k, v in obj.items()}

    if isinstance(obj, dict) and "data" in obj:
        out = {}
        for k, entry in obj["data"].items():
            if not isinstance(entry, dict) or "input" not in entry:
                continue
            out[normalize_entrez_key(k)] = _to_vec(entry["input"])
        return out

    raise ValueError(f"Unrecognized pickle format: {pkl_path}")

def attach_labels_to_embeddings(emb_dict: Dict[str, np.ndarray],
                                df_labels: pd.DataFrame,
                                entrez_to_col: Dict[str, str],
                                expected_dim: Optional[int]) -> Dict[str, Any]:
    # sanity on dims
    if expected_dim is not None:
        for k, v in emb_dict.items():
            if v.shape != (expected_dim,):
                raise ValueError(f"Key {k} has shape {v.shape}, expected ({expected_dim},)")

    cell_lines = df_labels["cell_line"].astype(str).tolist()
    out: Dict[str, Any] = {"_cell_lines": cell_lines, "data": {}}

    for entrez_id, vec in emb_dict.items():
        entry: Dict[str, Any] = {"input": vec}
        col = entrez_to_col.get(entrez_id)
        if col is not None:
            entry["label"] = df_labels[col].to_numpy(dtype=float, copy=True)
        out["data"][entrez_id] = entry
    return out

def load_labels_or_none(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        print(f"[warn] labels CSV not found, skipping: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if df.columns[0] != "cell_line":
        raise ValueError(f'First column in labels CSV must be "cell_line": {csv_path}')
    return df

def main():
    ap = argparse.ArgumentParser(description="Attach gene essentiality labels to Mahi embeddings.")
    ap.add_argument("--mahi_root", required=True,
                    help="Directory containing Mahi .pkl embedding files (e.g., .../mahi_embeddings/)")
    ap.add_argument("--data_dir", default="data",
                    help="Path to the data directory (default: ./data)")
    ap.add_argument("--expected_dim", type=int, default=None,
                    help="If set, enforce this embedding dimension")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite outputs if they already exist")
    args = ap.parse_args()

    tissue_root = Path(args.mahi_root)
    data_dir = Path(args.data_dir)
    expected_dim = args.expected_dim if args.expected_dim is not None else EXPECTED_DIM
    skip_if_exists = not args.overwrite

    if not tissue_root.exists():
        raise SystemExit(f"tissue_root does not exist: {tissue_root}")

    labels = {}
    for split in ("train", "test"):
        csv_path = data_dir / f"gene_essentiality_{split}.csv"
        df = load_labels_or_none(csv_path)
        if df is not None:
            labels[split] = {
                "df": df,
                "map": build_entrez_to_col(df),
                "csv": csv_path
            }

    if not labels:
        raise SystemExit(f"No label CSVs found in {data_dir} (looked for gene_essentiality_train.csv and gene_essentiality_test.csv)")

    pkls = sorted(
        p for p in tissue_root.iterdir()
        if p.is_file()
        and p.suffix == ".pkl"
        and not p.name.endswith(".with_labels.pkl")
    )
    if not pkls:
        raise SystemExit(f"No .pkl files found under {tissue_root}")

    print(f"[info] Found {len(pkls)} pickle files under {tissue_root}", flush=True)
    for split, obj in labels.items():
        print(f"[info] Using {split} labels: {obj['csv']}")

    totals = { "wrote": 0, "skipped": 0, "files": 0 }
    for pkl in pkls:
        totals["files"] += 1

        # load embeddings once per file, reuse for both splits
        try:
            emb_map = load_any_embedding_dict(pkl)
        except Exception as e:
            print(f"[ERR] {pkl.name} (load): {e}", flush=True)
            continue

        for split, obj in labels.items():
            out_pkl = pkl.with_name(pkl.stem + f".{split}.with_labels.pkl")
            if skip_if_exists and out_pkl.exists():
                print(f"[skip] {out_pkl.name} already exists", flush=True)
                totals["skipped"] += 1
                continue

            try:
                out_obj = attach_labels_to_embeddings(
                    emb_map, obj["df"], obj["map"], expected_dim
                )
                with open(out_pkl, "wb") as f:
                    pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

                n_emb = len(emb_map)
                n_labeled = sum(1 for e in out_obj["data"].values() if "label" in e)
                print(f"[OK] {pkl.name} → {out_pkl.name} | genes={n_emb} | with_labels={n_labeled}", flush=True)
                totals["wrote"] += 1
            except Exception as e:
                print(f"[ERR] {pkl.name} ({split}): {e}", flush=True)

    print(f"\n[summary] wrote={totals['wrote']} | skipped={totals['skipped']} | total_input_files={totals['files']}", flush=True)

if __name__ == "__main__":
    main()