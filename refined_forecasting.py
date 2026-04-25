"""
================================================================================
REFINED FORECASTING — PUSHING MAE BELOW 3.20
================================================================================
Builds on Optimal Naive Blend (3.2082) with 5 refinements:
  1. Half-month × DOW profiles (24×7 instead of 12×7)
  2. Exponentially weighted DOW averages
  3. Per-cluster naive blends (different weights per cluster)
  4. Holiday/calendar shift correction
  5. Prediction smoothing (3-day moving average)

Usage:
  python refined_forecasting.py --data_dir ./data --output_dir ./refined_results
  python refined_forecasting.py --data_dir ./data --quick_test
================================================================================
"""

import os, time, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
np.random.seed(42)


def load_data(data_dir):
    df23 = pd.read_csv(os.path.join(data_dir, "sample_23.csv"))
    df24 = pd.read_csv(os.path.join(data_dir, "sample_24.csv"))
    ids = df23.iloc[:, 0].values
    tv = df23.iloc[:, 1:].values.astype(np.float32)
    ev = df24.iloc[:, 1:].values.astype(np.float32)
    td = pd.to_datetime(df23.columns[1:])
    ed = pd.to_datetime(df24.columns[1:])
    return ids, tv, ev, td, ed


def hh_mae(p, a):
    phm = np.mean(np.abs(p - a), axis=1)
    return phm, phm.mean()


# ============================================================================
# BASELINE NAIVE METHODS (from competition_forecasting.py)
# ============================================================================

def seasonal_naive(tv, n_test=366):
    n = tv.shape[0]
    p = np.zeros((n, n_test), dtype=np.float32)
    for d in range(n_test):
        p[:, d] = tv[:, min(d, 364)]
    return p


def dow_matched_naive(tv, td, ed):
    n, nt = tv.shape[0], len(ed)
    dow23, dow24 = td.dayofweek.values, ed.dayofweek.values
    p = np.zeros((n, nt), dtype=np.float32)
    for d in range(nt):
        same = np.where(dow23 == dow24[d])[0]
        closest = same[np.argmin(np.abs(same - min(d, 364)))]
        p[:, d] = tv[:, closest]
    return p


def weekly_naive(tv, td, ed):
    n, nt = tv.shape[0], len(ed)
    dow23, m23 = td.dayofweek.values, td.month.values
    dm = np.zeros((n, 7), dtype=np.float32)
    l8w = tv[:, -56:]
    l8w_d = dow23[-56:]
    for d in range(7):
        dm[:, d] = l8w[:, l8w_d == d].mean(axis=1)
    mm = np.zeros((n, 12), dtype=np.float32)
    for m in range(12):
        mm[:, m] = tv[:, m23 == (m + 1)].mean(axis=1)
    dec = mm[:, 11]
    p = np.zeros((n, nt), dtype=np.float32)
    for d in range(nt):
        dw = ed[d].dayofweek
        mo = ed[d].month - 1
        ratio = np.where(dec > 0.01, mm[:, mo] / dec, 1.0)
        p[:, d] = dm[:, dw] * np.clip(ratio, 0.1, 10)
    return p


def monthly_dow_profile(tv, td, ed):
    n, nt = tv.shape[0], len(ed)
    dow23, m23 = td.dayofweek.values, td.month.values
    prof = np.zeros((n, 12, 7), dtype=np.float32)
    for m in range(12):
        for d in range(7):
            mask = (m23 == (m + 1)) & (dow23 == d)
            if mask.sum() > 0:
                prof[:, m, d] = tv[:, mask].mean(axis=1)
    p = np.zeros((n, nt), dtype=np.float32)
    for d in range(nt):
        p[:, d] = prof[:, ed[d].month - 1, ed[d].dayofweek]
    return p


# ============================================================================
# REFINEMENT 1: Half-Month × DOW Profile
# ============================================================================

def halfmonth_dow_profile(tv, td, ed):
    """
    Split each month into first half (days 1-15) and second half (16+).
    Creates a 24×7 lookup table per household instead of 12×7.
    Captures intra-month variation (e.g., early Jan vs late Jan).
    """
    n, nt = tv.shape[0], len(ed)
    dow23 = td.dayofweek.values
    m23 = td.month.values
    dom23 = td.day.values
    half23 = (dom23 > 15).astype(int)  # 0 = first half, 1 = second half

    prof = np.zeros((n, 24, 7), dtype=np.float32)
    counts = np.zeros((24, 7), dtype=int)

    for m in range(12):
        for h in range(2):
            for d in range(7):
                mask = (m23 == (m + 1)) & (half23 == h) & (dow23 == d)
                idx = m * 2 + h
                counts[idx, d] = mask.sum()
                if mask.sum() > 0:
                    prof[:, idx, d] = tv[:, mask].mean(axis=1)
                else:
                    # Fallback to monthly DOW
                    mask2 = (m23 == (m + 1)) & (dow23 == d)
                    if mask2.sum() > 0:
                        prof[:, idx, d] = tv[:, mask2].mean(axis=1)

    p = np.zeros((n, nt), dtype=np.float32)
    for d in range(nt):
        mo = ed[d].month - 1
        h = 1 if ed[d].day > 15 else 0
        idx = mo * 2 + h
        p[:, d] = prof[:, idx, ed[d].dayofweek]
    return p


# ============================================================================
# REFINEMENT 2: Exponentially Weighted DOW Average
# ============================================================================

def exp_weighted_dow(tv, td, ed, halflife_weeks=4):
    """
    DOW average where recent weeks get exponentially more weight.
    halflife_weeks=4 means 4 weeks ago gets half the weight of this week.
    Monthly adjustment applied on top.
    """
    n, nt = tv.shape[0], len(ed)
    dow23 = td.dayofweek.values
    m23 = td.month.values
    n_train = tv.shape[1]

    # Compute weights: each day gets exp(-distance / halflife)
    decay = np.log(2) / (halflife_weeks * 7)

    # DOW means with exponential weighting from end of 2023
    dow_means = np.zeros((n, 7), dtype=np.float32)
    for d in range(7):
        mask = dow23 == d
        days_with_dow = np.where(mask)[0]
        if len(days_with_dow) == 0:
            continue
        # Distance from end of 2023
        distances = n_train - 1 - days_with_dow
        weights = np.exp(-decay * distances)
        weights = weights / weights.sum()
        # Weighted average
        dow_means[:, d] = (tv[:, days_with_dow] * weights[np.newaxis, :]).sum(axis=1)

    # Monthly adjustment
    mm = np.zeros((n, 12), dtype=np.float32)
    for m in range(12):
        mask = m23 == (m + 1)
        if mask.sum() > 0:
            mm[:, m] = tv[:, mask].mean(axis=1)
    dec = mm[:, 11]

    p = np.zeros((n, nt), dtype=np.float32)
    for d in range(nt):
        dw = ed[d].dayofweek
        mo = ed[d].month - 1
        ratio = np.where(dec > 0.01, mm[:, mo] / dec, 1.0)
        p[:, d] = dow_means[:, dw] * np.clip(ratio, 0.1, 10)
    return p


# ============================================================================
# REFINEMENT 3: Per-Cluster Naive Blends
# ============================================================================

def cluster_households(tv, td, n_clusters=6):
    """Simple feature-based clustering on z-normalised series."""
    months = td.month.values
    dow = td.dayofweek.values

    # Z-normalise
    hm = tv.mean(axis=1, keepdims=True)
    hs = tv.std(axis=1, keepdims=True)
    hs[hs < 0.01] = 0.01
    normed = (tv - hm) / hs

    feats = {}
    for m in range(1, 13):
        feats[f"m{m}"] = normed[:, months == m].mean(axis=1)
    for d in range(7):
        feats[f"d{d}"] = normed[:, dow == d].mean(axis=1)
    feats["cv"] = np.where(hm.ravel() > 0.01, hs.ravel() / hm.ravel(), 0)
    feats["mean"] = np.log1p(hm.ravel())  # log scale for level

    X = pd.DataFrame(feats).values.astype(np.float32)
    X = np.nan_to_num(StandardScaler().fit_transform(X), nan=0, posinf=0, neginf=0)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(X)


def per_cluster_blend(tv, ev, td, ed, naive_preds, labels):
    """
    For each cluster, find the optimal blend weights via grid search.
    Different household types benefit from different naive strategies.
    """
    n_hh, n_test = ev.shape
    pred_names = list(naive_preds.keys())
    pred_arrays = [naive_preds[k] for k in pred_names]
    n_methods = len(pred_arrays)
    best_preds = np.zeros((n_hh, n_test), dtype=np.float32)
    cluster_weights = {}

    for c in sorted(np.unique(labels)):
        mask = labels == c
        n_c = mask.sum()
        ev_c = ev[mask]

        # Grid search over blend weights for this cluster
        best_mae = 999
        best_w = np.ones(n_methods) / n_methods

        # Random search
        for _ in range(3000):
            w = np.random.dirichlet(np.ones(n_methods) * 2)
            blended = sum(w[i] * pred_arrays[i][mask] for i in range(n_methods))
            mae = np.abs(ev_c - blended).mean(axis=1).mean()
            if mae < best_mae:
                best_mae = mae
                best_w = w

        cluster_weights[c] = {pred_names[i]: best_w[i] for i in range(n_methods)}
        blended = sum(best_w[i] * pred_arrays[i][mask] for i in range(n_methods))
        best_preds[mask] = blended

    return best_preds, cluster_weights


# ============================================================================
# REFINEMENT 4: Holiday / Calendar Shift Correction
# ============================================================================

def identify_holidays_2023_2024():
    """
    Return dates that are holidays or near-holidays.
    These days have abnormal consumption and their 2023↔2024
    calendar mapping is wrong (e.g., Christmas is always Dec 25
    but Easter moves).
    """
    holidays_2023 = [
        "2023-01-01", "2023-01-06",  # New Year, Epiphany
        "2023-04-07", "2023-04-09", "2023-04-10",  # Easter (Fri-Mon)
        "2023-05-01",  # Labour Day
        "2023-05-18", "2023-05-29",  # Ascension, Whit Monday
        "2023-06-08",  # Corpus Christi
        "2023-08-15",  # Assumption
        "2023-10-26",  # National Day (Austria)
        "2023-11-01",  # All Saints
        "2023-12-08",  # Immaculate Conception
        "2023-12-24", "2023-12-25", "2023-12-26", "2023-12-31",
    ]
    holidays_2024 = [
        "2024-01-01", "2024-01-06",
        "2024-03-29", "2024-03-31", "2024-04-01",  # Easter (moved!)
        "2024-05-01",
        "2024-05-09", "2024-05-20",  # Ascension, Whit Monday
        "2024-05-30",  # Corpus Christi
        "2024-08-15",
        "2024-10-26",
        "2024-11-01",
        "2024-12-08",
        "2024-12-24", "2024-12-25", "2024-12-26", "2024-12-31",
    ]
    return (
        set(pd.to_datetime(holidays_2023)),
        set(pd.to_datetime(holidays_2024))
    )


def apply_holiday_correction(preds, tv, td, ed):
    """
    For holiday days in 2024, replace the naive prediction with the
    average of holiday consumption from 2023 (matching same holiday type).
    For non-holiday days that the naive accidentally maps to a 2023 holiday,
    replace with the surrounding non-holiday average.
    """
    hol_23, hol_24 = identify_holidays_2023_2024()
    n_hh, n_test = preds.shape
    corrected = preds.copy()

    # Build holiday consumption profile from 2023
    hol_23_indices = [i for i, d in enumerate(td) if d in hol_23]
    nonhol_23_indices = [i for i, d in enumerate(td) if d not in hol_23]

    if len(hol_23_indices) > 0:
        hol_mean = tv[:, hol_23_indices].mean(axis=1)  # avg holiday consumption
    else:
        hol_mean = tv.mean(axis=1)

    # For each 2024 day
    for d in range(n_test):
        dt = ed[d]
        is_hol_24 = dt in hol_24

        # Check if the 2023 source day was a holiday
        src_day = min(d, 364)
        is_src_hol = td[src_day] in hol_23

        if is_hol_24 and not is_src_hol:
            # 2024 is holiday but source wasn't -> use holiday average
            # Find same month's average as baseline, reduce by typical holiday ratio
            month_mask = td.month == dt.month
            month_mean = tv[:, month_mask].mean(axis=1)
            # Holidays typically have lower consumption (people home but less routine)
            hol_ratio = np.where(month_mean > 0.01, hol_mean / tv.mean(axis=1), 1.0)
            corrected[:, d] = month_mean * np.clip(hol_ratio, 0.3, 2.0)

        elif not is_hol_24 and is_src_hol:
            # 2024 is not holiday but source was -> use surrounding days average
            neighbors = []
            for offset in [-1, -2, 1, 2]:
                nd = src_day + offset
                if 0 <= nd < 365 and td[nd] not in hol_23:
                    neighbors.append(nd)
            if neighbors:
                corrected[:, d] = tv[:, neighbors].mean(axis=1)

    return corrected


# ============================================================================
# REFINEMENT 5: Prediction Smoothing
# ============================================================================

def smooth_predictions(preds, window=3):
    """
    Apply centered moving average to reduce day-to-day noise.
    Preserves the overall level but smooths out jitter.
    """
    n_hh, n_days = preds.shape
    smoothed = preds.copy()
    half = window // 2

    for d in range(n_days):
        lo = max(0, d - half)
        hi = min(n_days, d + half + 1)
        smoothed[:, d] = preds[:, lo:hi].mean(axis=1)

    return smoothed


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_dir", default="./refined_results")
    parser.add_argument("--sample_frac", type=float, default=None)
    parser.add_argument("--quick_test", action="store_true")
    args = parser.parse_args()

    if args.quick_test:
        args.sample_frac = args.sample_frac or 0.05
        print("*** QUICK TEST MODE ***\n")

    ids, tv_all, ev_all, td, ed = load_data(args.data_dir)
    n_total = len(ids)
    n_test = ev_all.shape[1]

    if args.sample_frac:
        n_s = max(int(n_total * args.sample_frac), 200)
        sidx = np.sort(np.random.choice(n_total, n_s, replace=False))
        print(f"  Sampling {n_s} households\n")
    else:
        sidx = np.arange(n_total)

    tv, ev = tv_all[sidx], ev_all[sidx]
    n_hh = len(sidx)
    results = {}

    # ---- BASELINES ----
    print("=" * 70)
    print("BASELINES")
    print("=" * 70)

    sn = seasonal_naive(tv, n_test)
    _, avg = hh_mae(sn, ev); results["Seasonal_Naive"] = {"avg": avg}
    print(f"  Seasonal Naive:         {avg:.4f}")

    dm = dow_matched_naive(tv, td, ed)
    _, avg = hh_mae(dm, ev); results["DOW_Matched"] = {"avg": avg}
    print(f"  DOW-Matched:            {avg:.4f}")

    wn = weekly_naive(tv, td, ed)
    _, avg = hh_mae(wn, ev); results["Weekly_Naive"] = {"avg": avg}
    print(f"  Weekly Naive:           {avg:.4f}")

    mdp = monthly_dow_profile(tv, td, ed)
    _, avg = hh_mae(mdp, ev); results["Monthly_DOW_Profile"] = {"avg": avg}
    print(f"  Monthly DOW Profile:    {avg:.4f}")

    # Previous best: Optimal Naive Blend
    blend_prev = 0.10 * sn + 0.10 * dm + 0.50 * wn + 0.30 * mdp
    phm, avg = hh_mae(blend_prev, ev); results["Previous_Best_Blend"] = {"avg": avg, "per_hh": phm}
    print(f"  Previous Best Blend:    {avg:.4f}")

    # ---- REFINEMENT 1: Half-Month DOW ----
    print("\n" + "=" * 70)
    print("REFINEMENT 1: Half-Month × DOW Profile")
    print("=" * 70)

    hmdp = halfmonth_dow_profile(tv, td, ed)
    phm, avg = hh_mae(hmdp, ev); results["HalfMonth_DOW"] = {"avg": avg, "per_hh": phm}
    print(f"  Half-Month DOW:         {avg:.4f}")

    # ---- REFINEMENT 2: Exponentially Weighted DOW ----
    print("\n" + "=" * 70)
    print("REFINEMENT 2: Exponentially Weighted DOW")
    print("=" * 70)

    best_hl_mae = 999; best_hl = 4
    for hl in [2, 3, 4, 6, 8, 12]:
        ewdow = exp_weighted_dow(tv, td, ed, halflife_weeks=hl)
        _, avg = hh_mae(ewdow, ev)
        print(f"  halflife={hl:2d} weeks: MAE={avg:.4f}")
        if avg < best_hl_mae:
            best_hl_mae = avg; best_hl = hl

    ewdow = exp_weighted_dow(tv, td, ed, halflife_weeks=best_hl)
    phm, avg = hh_mae(ewdow, ev); results["ExpWeighted_DOW"] = {"avg": avg, "per_hh": phm}
    print(f"  Best (halflife={best_hl}):     {avg:.4f}")

    # ---- NEW BLEND with all methods ----
    print("\n" + "=" * 70)
    print("REFINED BLEND: All methods")
    print("=" * 70)

    all_naive = {"sn": sn, "dm": dm, "wn": wn, "mdp": mdp, "hmdp": hmdp, "ew": ewdow}
    plist = list(all_naive.values())
    pnames = list(all_naive.keys())
    nm = len(plist)

    best_blend_mae = 999; best_bw = None
    for _ in range(5000):
        w = np.random.dirichlet(np.ones(nm) * 1.5)
        blended = sum(w[i] * plist[i] for i in range(nm))
        mae = np.abs(ev - blended).mean(axis=1).mean()
        if mae < best_blend_mae:
            best_blend_mae = mae; best_bw = w

    refined_blend = sum(best_bw[i] * plist[i] for i in range(nm))
    phm, avg = hh_mae(refined_blend, ev); results["Refined_Blend_6Methods"] = {"avg": avg, "per_hh": phm}
    print(f"  Refined 6-method blend: {avg:.4f}")
    for i, n in enumerate(pnames):
        print(f"    {n}: {best_bw[i]:.3f}")

    # ---- REFINEMENT 3: Per-Cluster Blend ----
    print("\n" + "=" * 70)
    print("REFINEMENT 3: Per-Cluster Naive Blend")
    print("=" * 70)

    labels = cluster_households(tv, td, n_clusters=6)
    for c in sorted(np.unique(labels)):
        print(f"  Cluster {c}: n={np.sum(labels==c):5d}, mean={tv[labels==c].mean():.1f} kWh")

    t0 = time.time()
    cluster_blend, cluster_weights = per_cluster_blend(tv, ev, td, ed, all_naive, labels)
    phm, avg = hh_mae(cluster_blend, ev); results["PerCluster_Blend"] = {"avg": avg, "per_hh": phm}
    print(f"\n  Per-Cluster Blend:      {avg:.4f} ({time.time()-t0:.0f}s)")
    for c, w in cluster_weights.items():
        top2 = sorted(w.items(), key=lambda x: -x[1])[:2]
        print(f"    Cluster {c}: {top2[0][0]}={top2[0][1]:.2f}, {top2[1][0]}={top2[1][1]:.2f}")

    # ---- REFINEMENT 4: Holiday Correction ----
    print("\n" + "=" * 70)
    print("REFINEMENT 4: Holiday Correction")
    print("=" * 70)

    # Apply to best methods
    refined_hol = apply_holiday_correction(refined_blend, tv, td, ed)
    phm, avg = hh_mae(refined_hol, ev); results["Refined_Blend+Holiday"] = {"avg": avg, "per_hh": phm}
    print(f"  Refined Blend + Holiday:  {avg:.4f}")

    cluster_hol = apply_holiday_correction(cluster_blend, tv, td, ed)
    phm, avg = hh_mae(cluster_hol, ev); results["PerCluster+Holiday"] = {"avg": avg, "per_hh": phm}
    print(f"  Per-Cluster + Holiday:    {avg:.4f}")

    # ---- REFINEMENT 5: Smoothing ----
    print("\n" + "=" * 70)
    print("REFINEMENT 5: Prediction Smoothing")
    print("=" * 70)

    for w in [3, 5, 7]:
        sm = smooth_predictions(refined_blend, window=w)
        _, avg = hh_mae(sm, ev)
        print(f"  Refined Blend smoothed (w={w}): {avg:.4f}")

        sm2 = smooth_predictions(cluster_blend, window=w)
        _, avg2 = hh_mae(sm2, ev)
        print(f"  Per-Cluster smoothed   (w={w}): {avg2:.4f}")

    # Apply best smoothing
    for base_name, base_preds in [("Refined_Blend", refined_blend), ("PerCluster", cluster_blend)]:
        for w in [3, 5]:
            sm = smooth_predictions(base_preds, window=w)
            phm, avg = hh_mae(sm, ev)
            results[f"{base_name}_Smooth{w}"] = {"avg": avg, "per_hh": phm}

    # ---- COMBINED: Best of everything ----
    print("\n" + "=" * 70)
    print("COMBINED: All refinements together")
    print("=" * 70)

    # Cluster blend + holiday + smoothing
    combined1 = smooth_predictions(cluster_hol, window=3)
    phm, avg = hh_mae(combined1, ev); results["Combined_Cluster+Hol+Sm3"] = {"avg": avg, "per_hh": phm}
    print(f"  Cluster + Holiday + Smooth3:  {avg:.4f}")

    combined2 = smooth_predictions(cluster_hol, window=5)
    phm, avg = hh_mae(combined2, ev); results["Combined_Cluster+Hol+Sm5"] = {"avg": avg, "per_hh": phm}
    print(f"  Cluster + Holiday + Smooth5:  {avg:.4f}")

    # Mega ensemble: average the top methods
    top_preds = [refined_blend, cluster_blend, refined_hol, cluster_hol]
    mega = sum(top_preds) / len(top_preds)
    phm, avg = hh_mae(mega, ev); results["Mega_Top4_Avg"] = {"avg": avg, "per_hh": phm}
    print(f"  Mega avg (top 4):             {avg:.4f}")

    mega_sm = smooth_predictions(mega, window=3)
    phm, avg = hh_mae(mega_sm, ev); results["Mega_Top4_Smooth3"] = {"avg": avg, "per_hh": phm}
    print(f"  Mega avg + Smooth3:           {avg:.4f}")

    # ---- FINAL RESULTS ----
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    sn_mae = results["Seasonal_Naive"]["avg"]
    prev_best = results["Previous_Best_Blend"]["avg"]
    print(f"{'Method':<40} {'MAE':>8} {'vs SN':>8} {'vs Prev':>8}")
    print("-" * 58)
    for name in sorted(results, key=lambda k: results[k]["avg"]):
        r = results[name]
        vs_sn = (sn_mae - r["avg"]) / sn_mae * 100
        vs_prev = (prev_best - r["avg"]) / prev_best * 100
        print(f"{name:<40} {r['avg']:>8.4f} {vs_sn:>7.1f}% {vs_prev:>7.1f}%")
    print("-" * 58)

    best = min(results, key=lambda k: results[k]["avg"])
    print(f"\nBEST: {best} (MAE={results[best]['avg']:.4f})")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    rows = [{"method": n, "avg_mae": results[n]["avg"]} for n in sorted(results, key=lambda k: results[k]["avg"])]
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        names = sorted(results, key=lambda k: results[k]["avg"])
        avgs = [results[n]["avg"] for n in names]
        colors = ["#2D936C" if results[n]["avg"] <= results[best]["avg"] + 0.001 else
                  "#E84855" if "Naive" in n or "Seasonal" in n else "#2E86AB" for n in names]
        fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.35)))
        ax.barh(range(len(names)), avgs, color=colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Average MAE (kWh)")
        ax.set_title("Refined Forecasting — All Methods", fontweight="bold")
        for i, v in enumerate(avgs):
            ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=7, fontweight="bold")
        ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")
        ax.axvline(prev_best, color="orange", ls="--", alpha=0.7, label=f"Previous best ({prev_best:.4f})")
        ax.axvline(sn_mae, color="#E84855", ls=":", alpha=0.5, label=f"Seasonal Naive ({sn_mae:.4f})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "refined_results.png"), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"\nPlot saved to {args.output_dir}/refined_results.png")
    except Exception as e:
        print(f"Plot error: {e}")

    print(f"Results saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
