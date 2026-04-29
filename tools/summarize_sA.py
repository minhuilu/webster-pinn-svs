# file: tools/summarize_sA.py
import argparse
import os
import sys
import pandas as pd
import numpy as np

DEFAULT_CSV = "exp/A_static/ckps/full_eval_all_sA.csv"

def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def build_tables(df: pd.DataFrame) -> str:
    # Keep only the best row for each (mode, run_id, tag), using minimum mSTFT_raw.
    need_cols = ["mode","run_id","tag","mSTFT_raw","LSD_raw_dB",
                 "mSTFT_aln","LSD_aln_dB","hnr_db_hat"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"CSV is missing a required column: {c}")

    # Force numeric columns to avoid mixed strings
    for c in ["mSTFT_raw","LSD_raw_dB","mSTFT_aln","LSD_aln_dB","hnr_db_hat"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with NaN in key metrics
    df = df.dropna(subset=["mSTFT_raw","LSD_raw_dB","mSTFT_aln","LSD_aln_dB","hnr_db_hat"]).copy()

    idx = df.groupby(["mode","run_id","tag"])["mSTFT_raw"].idxmin()
    best = df.loc[idx].copy()

    # Aggregate a/i/u averages for each (mode, run_id)
    agg_raw = best.groupby(["mode","run_id"], as_index=False).agg(
        mSTFT_raw_mean = ("mSTFT_raw","mean"),
        LSD_raw_dB_mean= ("LSD_raw_dB","mean"),
        HNR_hat_mean   = ("hnr_db_hat","mean"),
    )
    agg_aln = best.groupby(["mode","run_id"], as_index=False).agg(
        mSTFT_aln_mean   = ("mSTFT_aln","mean"),
        LSD_aln_dB_mean  = ("LSD_aln_dB","mean"),
        HNR_hat_mean     = ("hnr_db_hat","mean"),
    )

    # Sort
    agg_raw  = agg_raw.sort_values(by="mSTFT_raw_mean").reset_index(drop=True)
    agg_aln  = agg_aln.sort_values(by="mSTFT_aln_mean").reset_index(drop=True)

    # Generate Markdown
    lines = []

    # Raw table
    lines.append("| Mode | Run | mSTFT↓ | LSD(dB)↓ | HNR(dB)↑ |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in agg_raw.iterrows():
        lines.append(
            f"| {r['mode']} | {r['run_id']} | "
            f"{fmt(r['mSTFT_raw_mean'],3)} | {fmt(r['LSD_raw_dB_mean'],2)} | {fmt(r['HNR_hat_mean'],2)} |"
        )

    # Blank line plus aligned table
    lines.append("")
    lines.append("(Aligned) | Mode | Run | mSTFT↓ | LSD(dB)↓ | HNR(dB)↑ |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in agg_aln.iterrows():
        lines.append(
            f"| {r['mode']} | {r['run_id']} | "
            f"{fmt(r['mSTFT_aln_mean'],3)} | {fmt(r['LSD_aln_dB_mean'],2)} | {fmt(r['HNR_hat_mean'],2)} |"
        )

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Summarize sA results to Markdown tables.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV,
                        help=f"Path to merged CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--out", type=str, default="",
                        help="Optional path to save the Markdown output (e.g., exp/A_static/ckps/summary.md)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found:{args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    md = build_tables(df)

    # Print to console
    print(md)

    # Optionally save to file
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"\n[OK] Markdown saved -> {args.out}")

if __name__ == "__main__":
    main()
