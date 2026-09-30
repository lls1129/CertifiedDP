"""Turn pe_means_table.csv into PE-means' Table 1: AUC of the normalized loss over eps (trapezoid rule, their eq. 7;
the non-private row is its loss times the eps range), one column per method, plus the baseline columns as reported
in their paper (Google-LSH, FastLloyd, DP-Lib, Icml17 -- not rerun here).  Also a label-accuracy table.

python library/examples/pe_means_table_summary.py results/pe_means_table.csv > results/pe_means_table.md"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from library import datasets, metrics

# arXiv 2606.00342 Table 1, columns Non-Priv, PE-means, HDPE-means, Google-LSH, FastLloyd, DP-Lib, Icml17
REPORTED = {
    "birch2": (0.0001, 0.0003, None, 0.0658, 0.0031, 0.0124, 0.0161), "iris": (0.1361, 0.2894, None, 0.6892, 0.3979, 1.3048, 5.5766),
    "adult": (0.0056, 0.0056, None, 0.0076, 0.0155, 0.0213, 0.0113), "mnist": (0.1958, 0.2920, 0.2169, 0.2173, 0.5822, 0.5315, 0.7904),
    "letter": (0.2481, 0.2852, None, 0.3187, 0.3963, 0.4341, 0.6277), "gas": (0.1026, 0.1081, None, 0.1121, 0.1561, 0.1761, 0.1853),
    "g2_4": (0.4650, 0.4761, None, 0.5465, 0.4666, 0.5468, 0.8802), "g2_16": (0.7677, 0.7800, None, 0.8391, 0.8687, 1.1816, 4.7489),
    "g2_64": (1.0956, 1.1538, 1.1295, 1.2478, 1.9647, 2.6608, 27.5361), "g2_128": (1.3534, 1.7237, 1.4055, 1.5906, 2.6425, 4.4632, 74.0978),
    "scale_4_4": (0.0821, 0.0814, None, 0.1190, 0.1272, 0.1381, 0.1772), "scale_4_16": (0.3013, 0.3147, None, 0.3392, 0.3721, 0.4114, 0.5948),
    "scale_4_64": (0.4966, 0.5326, 0.5340, 0.5430, 0.5863, 0.6622, 2.5354), "scale_4_128": (0.5731, 0.6179, 0.5940, 0.5987, 0.6282, 0.7481, 8.5885),
    "scale_16_4": (0.0302, 0.0335, None, 0.0555, 0.0864, 0.0976, 0.1525), "scale_16_16": (0.2110, 0.2452, None, 0.2880, 0.3624, 0.3778, 0.5679),
    "scale_16_64": (0.4824, 0.5884, 0.5757, 0.5823, 0.6371, 0.6949, 2.5353), "scale_16_128": (0.5410, 0.6131, 0.6023, 0.5992, 0.6294, 0.7612, 8.6815),
    "scale_64_4": (0.0029, 0.0051, None, 0.0214, 0.0306, 0.0381, 0.0847), "scale_64_16": (0.1126, 0.1607, None, 0.1902, 0.2393, 0.2771, 0.5238),
    "scale_64_64": (0.4487, 0.5518, 0.5567, 0.5453, 0.5827, 0.6913, 2.5620), "scale_64_128": (0.5206, 0.6874, 0.6232, 0.6299, 0.6421, 1.0248, 8.6433),
    "sklearn_4_4": (0.0777, 0.0778, None, 0.1305, 0.1777, 0.2355, 0.1240), "sklearn_16_4": (0.0459, 0.0445, None, 0.0979, 0.0990, 0.1409, 0.2601),
    "sklearn_64_4": (0.0343, 0.0379, None, 0.1289, 0.0818, 0.1191, 0.3227), "sklearn_4_16": (0.0950, 0.0955, None, 0.1395, 0.7110, 0.5049, 0.6074),
    "sklearn_16_16": (0.0606, 0.0668, None, 0.1596, 0.7160, 0.5809, 1.4670), "sklearn_64_16": (0.0642, 0.1183, None, 0.3872, 0.7631, 0.8720, 1.7704),
    "sklearn_4_64": (0.0954, 0.0993, 0.0974, 0.1090, 1.4623, 0.8614, 2.7731), "sklearn_16_64": (0.0847, 0.1192, 0.1189, 0.2258, 1.7475, 1.3716, 3.1765),
    "sklearn_64_64": (0.0823, 0.3070, 0.4678, 0.7878, 2.2002, 2.2456, 3.1698), "sklearn_4_128": (0.1054, 0.1333, 0.1097, 0.1186, 1.7604, 1.1632, 5.6663),
    "sklearn_16_128": (0.0919, 0.1942, 0.1562, 0.2838, 2.3038, 1.9742, 4.9881), "sklearn_64_128": (0.0933, 0.5740, 0.7190, 0.9675, 2.7204, 2.9608, 4.9597),
}
REP_COLS = ["Non-Priv", "PE-means", "HDPE-means", "Google-LSH", "FastLloyd", "DP-Lib", "Icml17"]
OURS = ["nonpriv", "ours", "pe_means", "hdpe_means", "hdpe_means_repo"]


def fmt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def main(*paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["eps"] = df["eps"].astype(float)
    eps = [float(e) for e in sorted(df.loc[np.isfinite(df.eps), "eps"].unique())]
    span = eps[-1] - eps[0]
    rows, acc_rows = [], []
    for name in [n for n in datasets.NAMES if n in set(df.dataset)]:
        d = df[df.dataset == name]
        info = d.iloc[0]
        row = {"dataset": name, "N": int(info.N), "d": int(info.d), "k": int(info.k)}
        arow = dict(row)
        for m in OURS:
            s = d[d.method == m]
            if s.empty:
                row[m] = arow[m] = None
                continue
            if m == "nonpriv":
                row[m] = s.loss.mean() * span
                arow[m] = s.label_acc.mean() if s.label_acc.notna().any() else None
            else:
                g = s.groupby("eps").loss.mean().reindex(eps)
                row[m] = metrics.loss_auc(eps, g.values) if g.notna().all() else None
                a = s[s.eps == 1.0].label_acc
                arow[m] = a.mean() if a.notna().any() else None
            row[m + "_seeds"] = s.seed.nunique()
        rep = REPORTED.get(name, (None,) * 7)
        row.update({f"rep_{c}": v for c, v in zip(REP_COLS, rep)})
        rows.append(row)
        acc_rows.append(arow)
    seeds = sorted({r.get("ours_seeds") for r in rows if r.get("ours_seeds")})
    print(f"## Loss AUC over eps in {eps} (delta = 1/N^1.1), {seeds} seed(s) per cell; "
          "'reported' columns copied from arXiv 2606.00342 Table 1 (their run, 50 seeds)\n")
    hdr = ["dataset", "N", "d", "k", "Non-Priv", "ours (r_out=20)", "PE-means", "HDPE-means (Alg. 3)", "HDPE-means (their code)"] + [f"{c} (reported)" for c in REP_COLS]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        best = min(v for v in [r["ours"], r["pe_means"], r["hdpe_means"], r["hdpe_means_repo"]] if v is not None) if r["ours"] is not None else None
        cells = [r["dataset"], r["N"], r["d"], r["k"]] + \
                [fmt(r["nonpriv"])] + [("**" + fmt(r[m]) + "**") if r[m] is not None and r[m] == best else fmt(r[m])
                                        for m in ("ours", "pe_means", "hdpe_means", "hdpe_means_repo")] + [fmt(r[f"rep_{c}"]) for c in REP_COLS]
        print("| " + " | ".join(str(c) for c in cells) + " |")
    lab = [r for r in acc_rows if r.get("nonpriv") is not None]
    if lab:
        print("\n## Label accuracy at eps = 1 (majority label per centre), labelled datasets\n")
        print("| dataset | Non-Priv | ours | PE-means | HDPE-means (Alg. 3) | HDPE-means (their code) |\n|---|---|---|---|---|---|")
        for r in lab:
            print(f"| {r['dataset']} | " + " | ".join(fmt(r[m]) for m in OURS) + " |")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["results/pe_means_table.csv"]))
